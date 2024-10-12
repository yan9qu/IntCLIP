<div align="center">
  
# IntCLIP: Synergy of Sight and Semantics: Visual Intention Understanding with CLIP

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![Conference](http://img.shields.io/badge/ECCV-2024-6790AC.svg)](https://eccv2024.ecva.net/)
[![Paper](http://img.shields.io/badge/Paper-6720AC.svg)](https://marswhu.github.io/publications/files/ECCV24_IntCLIP.pdf)

</div>

## Updates

- :blush: (12/10/2024) Code Released!
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
pip install -r requirements.txt
cd ..
```

In the conda environment, install `pycocotools` and `randaugment` with pip:
```
pip install pycocotools
pip install randaugment
```

**Install extra dependencies**

Follow [the link](https://github.com/KaiyangZhou/Dassl.pytorch) to install `dassl`.

### Data preparation

**Prepare datasets**

warning: Please note that we do not own the copyrights of any datasets we used. Please contact the original authors to get access to the images, follow [the link](https://github.com/KMnP/intentonomy).

You should specify the following directory in opts.py:
```
--datadir
--dataset_config_file
```

### Training

Training IntCLIP on Intentonomy:

```bash
python train.py 
```

### Inference

The results of the test set and validation set, obtained from the output of the train.py file during training

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
 * DualCoOp [https://github.com/asyml/DualCoOp](https://github.com/sunxm2357/DualCoOp)
 * FameVIL [https://github.com/1Konny/mmf](https://github.com/BrandonHanx/mmf)
