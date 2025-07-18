<div align="center">
<h1>Large-Scale 3D Medical Image Pre-training with Geometric Context Priors</h1>

<a href="https://github.com/Luffy03/Large-Scale-Medical"><img src='https://img.shields.io/badge/arXiv-Preprint-red' alt='Paper PDF'></a>
<a href="https://openaccess.thecvf.com/content/CVPR2024/html/Wu_VoCo_A_Simple-yet-Effective_Volume_Contrastive_Learning_Framework_for_3D_Medical_CVPR_2024_paper.html"><img src='https://img.shields.io/badge/CVPR-Conference-red' alt='Paper PDF'></a>
</div>

**A simple baseline for [FLARE25_Task4](https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task4-CT-FM)**:

GitHub link: [https://github.com/Luffy03/Large-Scale-Medical](https://github.com/Luffy03/Large-Scale-Medical) (CVPR 2024 Extension)

Paper abstract: We observe that 3D medical images contain consistent geometric context, *i.e.*, consistent geometric relations between different organs, which leads to a promising way for learning consistent representations.
Motivated by this, we propose a simple-yet-effective **Vo**lume **Co**ntrast (**VoCo**) framework to leverage geometric context priors for self-supervision. 
Given an input volume, we extract base crops from different regions to construct positive and negative pairs for contrastive learning. Then we predict the contextual position of a random crop by contrasting its similarity to the base crops.
In this way, VoCo implicitly encodes the inherent geometric context into model representations, facilitating high-level semantic learning without annotations.

![framework](assets/framework.png)

## Pre-training

### Pre-trained model
[Google drive](https://drive.google.com/file/d/1U2HvC6_8TN71NN83Hoz8e-dPGbO_rtLC/view?usp=drive_link).

### Download Pre-training Dataset

The datasets of [FLARE25_Task4](https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task4-CT-FM) should be organized as:
```
├── FLARE-Task4-CT-FM
    ├── train_part1
    └── train_part2
```

### Usage

```bash
cd Self-supervised
source activate YOUR-CONDA-ENVIRONMENT
# single GPU, if you don't have enough gpu resource
sh single_train
# multi-gpu
sh dist_B.sh
sh dist_L.sh
sh dist_H.sh
```

## Downstream

The implementations of downstream tasks are available at [TASK4_Downstream](https://github.com/lwubfcse/FLARE25_Task4_baseline_VoCo/tree/main/TASK4_Downstream).
You only need to modify the path of downstream dataset at './data/$dataset_name', and run:
```bash
sh train.sh
```
Checkpoints and training logs as follows:

| Task               |                                                                                  Checkpoint |                                          Val (Dice/Acc)                                          |
|:-------------------|--------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|
| Lung_lesion_seg    | [Model](https://drive.google.com/drive/folders/1-ogVBP3F2LSpdvy8yhakDTGI3C2lwdgR?usp=drive_link) |                                             [60.03](https://drive.google.com/drive/folders/1-ogVBP3F2LSpdvy8yhakDTGI3C2lwdgR?usp=drive_link)                                              |
| Abdomen_lesion_seg |                                                                                         [Model](https://drive.google.com/drive/folders/1-ogVBP3F2LSpdvy8yhakDTGI3C2lwdgR?usp=drive_link) |                                             [41.99](https://drive.google.com/drive/folders/1-ogVBP3F2LSpdvy8yhakDTGI3C2lwdgR?usp=drive_link)                                              |
| Abdomen_organ_seg  |                                                                                        [Model](https://drive.google.com/drive/folders/1-ogVBP3F2LSpdvy8yhakDTGI3C2lwdgR?usp=drive_link) | [83.55](https://drive.google.com/drive/folders/1-ogVBP3F2LSpdvy8yhakDTGI3C2lwdgR?usp=drive_link) |

For more details, please refer to [VoCo](https://github.com/Luffy03/Large-Scale-Medical/tree/main/Downstream).


## Citation
**Please refer to [FLARE25 challenge](https://flare-medfm.github.io/).**

If you find this repo useful for your research, please consider citing the paper as follows:

```bibtex
@article{voco,
  title={Large-Scale 3D Medical Image Pre-training with Geometric Context Priors},
  author={Wu, Linshan and Zhuang, Jiaxin and Chen, Hao},
  journal={arXiv preprint arXiv:2410.09890},
  year={2024}
}
@InProceedings{voco-v1,
    author    = {Wu, Linshan and Zhuang, Jiaxin and Chen, Hao},
    title     = {VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis},
    booktitle = {CVPR},
    month     = {June},
    year      = {2024},
    pages     = {22873-22882}
}
```
