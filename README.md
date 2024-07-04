<div align="center">
  
# IntCLIP: Synergy of Sight and Semantics: Visual Intention Understanding with CLIP

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![Conference](http://img.shields.io/badge/ECCV-2024-6790AC.svg)](https://eccv2024.ecva.net/)
<!-- [![Paper](http://img.shields.io/badge/Paper-arxiv.2407-B31B1B.svg)](https://arxiv.org/)
[![Poster](http://img.shields.io/badge/Poster-5F5F5F.svg)]() -->

</div>

## Updates

- :blush: (02/07/2024) Paper Accepted!

## Abstract

![](assests/intro.png)

Multi-label Intention Understanding (MIU) for images is a critical yet challenging domain, primarily due to the ambiguity of intentions leading to a resource-intensive annotation process. Current leading approaches are held back by the limited amount of labeled data. To mitigate the scarcity of annotated data, we leverage the Contrastive Language-Image Pre-training (CLIP) model, renowned for its proficiency in aligning textual and visual modalities. We introduce a novel framework, **Intention Understanding with CLIP** (IntCLIP), which utilizes a dual-branch approach. This framework exploits the 'Sight'-oriented knowledge inherent in CLIP to augment 'Semantic'-centric MIU tasks. Additionally, we propose **Hierarchical Class Integration** to effectively manage the complex layered label structure, aligning it with CLIP's nuanced sentence feature extraction capabilities. Our **Sight-assisted Aggregation** further refines this model by infusing the semantic feature map with essential visual cues, thereby enhancing the intention understanding ability. Through extensive experiments conducted on the standard MIU benchmark and other subjective tasks such as Image Emotion Recognition, IntCLIP clearly demonstrates superiority over current state-of-the-art techniques.

## Architecture

![](assests/architecture.png)

**Overview of proposed Intention Understanding with CLIP (IntCLIP)**. **(a)** Demonstrates the MIU with CLIP process, incorporating Sight-assisted Aggregation to fuse sight and semantic features with text encoding, followed by Asymmetric Loss (AS Loss) optimization to align final image and text features. The Hierarchical Class Integration extracts abundant class information from hierarchical labels. **(b)** Depicts the Sight-semantic Image Encoding strategy, where the CLIP-initialized encoder is dedicated to capturing sight-related features, and the semantic branch, through partially learnable deep layers, evolves to accommodate semantic information, thereby preserving both objective visual details and subjective semantic insights.

## Getting started

### Dependencies installation

**Install IntCLIP from scratch**

```bash
conda create -n intclip python=3.7
conda activate intclip

git clone https://github.com/yan9qu/IntCLIP.git
cd IntCLIP
pip install --editable .
cd ..
```

**Install extra dependencies**

```bash
pip install wandb einops
```

### Data preparation

<!--**Download datasets**

:warning: Please note that we do not own the copyrights of any datasets we used. We can only share the pre-processed caption data here via [Google Drive](https://drive.google.com/drive/folders/1rmSfvKcTvVLugBWFNFtIDZuQwRPGr87U?usp=sharing). Please contact the original authors to get access to the images.

**Download pre-trained models**

You can download our pre-trained VQVAE via [Google Drive](https://drive.google.com/file/d/11QKoXEG1NeFqUyLg4kOjkJTgQsiYHpdu/view?usp=sharing).


### Training

Fine-tune pre-trained FashionViL on cross-modal retrieval on FashionGen:

```bash
python mmf_cli/run.py \
config=projects/fashionvil/configs/e2e_contrastive.yaml \
model=fashionvil \
dataset=fashiongen
```

Fine-tune pre-trained FashionViL on text-guided image retrieval on FashionIQ:

```bash
python mmf_cli/run.py \
config=projects/fashionvil/configs/e2e_composition.yaml \
model=fashionvil \
dataset=fashioniq
```

Fine-tune pre-trained FashionViL on sub-category classification on FashionGen:

```bash
python mmf_cli/run.py \
config=projects/fashionvil/configs/e2e_classification.yaml \
model=fashionvil \
dataset=fashiongen
```

Fine-tune pre-trained FashionViL on outfit complementary item retrieval on PolyvoreOutfits:

```bash
python mmf_cli/run.py \
config=projects/ocir/configs/polyvore/defaults.yaml \
model=csa_net \
dataset=polyvore_ocir
```


### Evaluation

Evaluate fine-tuned FashionViL on cross-modal retrieval on FashionGen:

```bash
python mmf_cli/run.py \
config=projects/fashionvil/configs/e2e_contrastive.yaml \
model=fashionvil \
dataset=fashiongen \
run_type=test \
checkpoint.resume_file=save/fashionvil_contrastive_fashiongen_e2e_pretrain_final/best.ckpt
```

Evaluate fine-tuned FashionViL on text-guided image retrieval on FashionIQ:

```bash
python mmf_cli/run.py \
config=projects/fashionvil/configs/e2e_composition.yaml \
model=fashionvil \
dataset=fashioniq \
run_type=test \
checkpoint.resume_file=save/fashionvil_composition_fashioniq_e2e_pretrain_final/best.ckpt
```

Evaluate fine-tuned FashionViL on sub-category classification on FashionGen:

```bash
python mmf_cli/run.py \
config=projects/fashionvil/configs/e2e_classification.yaml \
model=fashionvil \
dataset=fashiongen \
run_type=test \
checkpoint.resume_file=save/fashionvil_classification_fashiongen_e2e_sub/best.ckpt
```

Evaluate fine-tuned FashionViL on outfit complementary item retrieval on PolyvoreOutfits:

```bash
python mmf_cli/run.py \
config=projects/ocir/configs/polyvore/defaults.yaml \
model=csa_net \
dataset=polyvore_ocir \
run_type=test \
checkpoint.resume_file=save/polyvore_csa_disjoint_final/best.ckpt
```
-->
## Citation

If you find this project useful for your research, please use the following BibTeX entry.

```
@inproceedings{intclip,
    title={Synergy of Sight and Semantics: Visual Intention Understanding with CLIP},
    author={Qu Yang, Mang Ye and Dacheng Tao},
    booktitle={ECCV},
    year={2024}
}
```
---
### :heart: We thank the following users who open repositories on Github for us to build on for our experiments
 * DualCoOp [https://github.com/asyml/texar-pytorch](https://github.com/sunxm2357/DualCoOp)
 * FameVIL [https://github.com/1Konny/Beta-VAE](https://github.com/BrandonHanx/mmf)
