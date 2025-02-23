import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
import torch.nn.functional as F

_tokenizer = _Tokenizer()

__all__ = ['dualcoop', 'DualCoop']


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model_conv_proj(state_dict or model.state_dict(), cfg)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(
            dim=-1)] @ self.text_projection

        return x


class MLCPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx_pos = cfg.TRAINER.COOP_MLC.N_CTX_POS
        n_ctx_neg = cfg.TRAINER.COOP_MLC.N_CTX_NEG
        ctx_init_pos = cfg.TRAINER.COOP_MLC.POSITIVE_PROMPT_INIT.strip()
        ctx_init_neg = cfg.TRAINER.COOP_MLC.NEGATIVE_PROMPT_INIT.strip()
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            prompt_pos = clip.tokenize(ctx_init_pos)
            prompt_neg = clip.tokenize(ctx_init_neg)
            with torch.no_grad():
                embedding_pos = clip_model.token_embedding(
                    prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(
                    prompt_neg).type(dtype)
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if cfg.TRAINER.COOP_MLC.CSC:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # Random Initialization
            if cfg.TRAINER.COOP_MLC.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors_pos = torch.empty(
                    n_cls, n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(
                    n_cls, n_ctx_neg, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)

        print(f'Initial positive context: "{prompt_prefix_pos}"')
        print(f'Initial negative  context: "{prompt_prefix_neg}"')
        print(f"Number of positive context words (tokens): {n_ctx_pos}")
        print(f"Number of negative context words (tokens): {n_ctx_neg}")

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts_pos = [prompt_prefix_pos + " " +
                       name + "." for name in classnames]
        prompts_neg = [prompt_prefix_neg + " " +
                       name + "." for name in classnames]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
        for p_pos, p_neg in zip(prompts_pos, prompts_neg):
            tokenized_prompts_pos.append(clip.tokenize(p_pos))
            tokenized_prompts_neg.append(clip.tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(
                tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(
                tokenized_prompts_neg).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix_pos", embedding_pos[:, :1, :])
        self.register_buffer("token_suffix_pos",
                             embedding_pos[:, 1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :1, :])
        self.register_buffer("token_suffix_neg",
                             embedding_neg[:, 1 + n_ctx_neg:, :])

        self.n_cls = n_cls
        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        tokenized_prompts = torch.cat(
            [tokenized_prompts_neg, tokenized_prompts_pos], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.name_lens = name_lens

    def forward(self, cls_id=None):
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg

        if ctx_pos.dim() == 2:
            if cls_id is None:
                ctx_pos = ctx_pos.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_pos = ctx_pos.unsqueeze(0).expand(len(cls_id), -1, -1)
        else:
            if cls_id is not None:
                ctx_pos = ctx_pos[cls_id]

        if ctx_neg.dim() == 2:
            if cls_id is None:
                ctx_neg = ctx_neg.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_neg = ctx_neg.unsqueeze(0).expand(len(cls_id), -1, -1)
        else:
            if cls_id is not None:
                ctx_neg = ctx_neg[cls_id]

        if cls_id is None:
            prefix_pos = self.token_prefix_pos
            prefix_neg = self.token_prefix_neg
            suffix_pos = self.token_suffix_pos
            suffix_neg = self.token_suffix_neg
        else:
            prefix_pos = self.token_prefix_pos[cls_id]
            prefix_neg = self.token_prefix_neg[cls_id]
            suffix_pos = self.token_suffix_pos[cls_id]
            suffix_neg = self.token_suffix_neg[cls_id]

        prompts_pos = torch.cat(
            [
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts = torch.cat([prompts_neg, prompts_pos], dim=0)

        if cls_id is not None:
            tokenized_prompts_pos = self.tokenized_prompts[self.n_cls:][cls_id]
            tokenized_prompts_neg = self.tokenized_prompts[:self.n_cls][cls_id]
            tokenized_prompts = torch.cat(
                [tokenized_prompts_neg, tokenized_prompts_pos], dim=0)
        else:
            tokenized_prompts = self.tokenized_prompts

        return prompts, tokenized_prompts


class DualCoop(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.visual_encoder_type = cfg.MODEL.BACKBONE.NAME
        self.prompt_learner = MLCPromptLearner(cfg, classnames, clip_model)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = cfg.TRAINER.COOP_MLC.LS
        self.dtype = clip_model.dtype
        self.cfg = cfg

    def forward(self, image, cls_id=None):
        # get image and text features
        image_features, feat_for_orth = self.image_encoder(
            image.type(self.dtype))
        prompts, tokenized_prompts = self.prompt_learner(cls_id)
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # normalize features
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)
        image_features_norm = image_features / \
            image_features.norm(dim=1, keepdim=True)

        # Class-Specific Region Feature Aggregation
        output = 20 * F.conv1d(image_features_norm, text_features[:, :, None])
        b, c, _ = output.shape
        output_half = output[:,  c // 2:]
        w_half = F.softmax(output_half, dim=-1)
        w = torch.cat([w_half, w_half], dim=1)
        output = 5 * (output * w).sum(-1)

        b, c = output.shape

        # convert the shape of logits to [b, 2, num_class]
        logits = output.resize(b, 2, c//2)

        return logits

    def text_cat_tokenize(self, cls_id=None):
        # get image and text features
        prompts, tokenized_prompts = self.prompt_learner(cls_id)

        return tokenized_prompts

    @property
    def network_name(self):
        name = ''
        name += 'DualCoop-{}'.format(self.visual_encoder_type)
        return name

    def backbone_params(self):
        params = []
        for name, param in self.named_parameters():
            if "image_encoder" in name and "prompt_learner" not in name and 'attnpool' not in name:
                params.append(param)
        return params

    def intent_params(self):
        params = []
        for name, param in self.named_parameters():
            if "image_encoder" in name and "prompt_learner" not in name and 'attnpool' not in name:
                if 'intent_layer3' in name or 'intent_layer4' in name:
                    params.append(param)
        return params

    def aggregation_params(self):
        params = []
        for name, param in self.named_parameters():
            if "position_aware_aggregation" in name:
                params.append(param)
        return params

    def attn_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'attnpool' in name and 'image_encoder' in name:
                params.append(param)
                print(name)
        return params

    def prompt_params(self):
        params = []
        for name, param in self.named_parameters():
            if "prompt_learner" in name:
                params.append(param)
        return params


def dualcoop(cfg, classnames, **kwargs):
    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
    clip_model = load_clip_to_cpu(cfg)

    clip_model.float()

    print("Building dualcoop")
    model = DualCoop(cfg, classnames, clip_model)

    if not cfg.TRAINER.FINETUNE_BACKBONE:
        print('Freeze the backbone weights')
        backbone_params = model.backbone_params()
        for param in backbone_params:
            param.requires_grad_(False)

    if not cfg.TRAINER.FINETUNE_ATTN:
        print('Freeze the attn weights')
        attn_params = model.attn_params()
        for param in attn_params:
            param.requires_grad_(False)

    if cfg.TRAINER.FINETUNE_LAST_LAYER:
        print('Unfreeze the 3,4 layers of resnet')
        intent_params = model.intent_params()
        for param in intent_params:
            param.requires_grad_(True)

    if cfg.TRAINER.FINETUNE_LAST_LAYER:
        print('Unfreeze the 3,4 layers of resnet')
        aggregation_params = model.aggregation_params()
        for param in aggregation_params:
            param.requires_grad_(True)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Note that multi-gpu training could be slow because CLIP's size is
    # big, which slows down the copy operation in DataParallel
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(
            f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        model = nn.DataParallel(model)
    return model
