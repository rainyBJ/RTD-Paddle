## 1 Introduction

This project reproduces RTD based on paddlepaddle framework. RTD is a simple and end-to-end learnable framework (RTD-Net) for direct action proposal generation, by re-purposing a Transformer-alike architecture. Thanks to the parallel decoding of multiple proposals with explicit context modeling, our RTD-Net outperforms the previous state-of-the-art methods in temporal action proposal generation task on THUMOS14 and also yields a superior performance for action detection on this dataset. In addition, free of NMS post-processing, our detection pipeline is more efficient than previous methods.

### Paper

[RTD-Net (ICCV 2021)](https://arxiv.org/pdf/2102.01894.pdf)

 "Relaxed Transformer Decoders for Direct Action Proposal Generation", accepted in ICCV 2021.

### Reference project

https://github.com/MCG-NJU/RTD-Action

### **The link of aistudio：**

https://aistudio.baidu.com/aistudio/projectdetail/2470811

## 2 Accuracy

| Dataset  | AR@50 | AR@100 | AR@200 | AR@500 | checkpoint                                                   |
| -------- | ----- | ------ | ------ | ------ | ------------------------------------------------------------ |
| THUMOS14 | 41.14 | 49.49  | 56.46  | 62.90  | [link](https://drive.google.com/file/d/1h20GnPhaJP3QkwVspn_ndXevJ97FGpE6/view?usp=sharing) |

## 3 Dataset

To reproduce the results in THUMOS14 without further changes:

1. Download the data from [GoogleDrive](https://drive.google.com/drive/folders/13KwgSgeZKWwIYE77PVo4_dvZhf8qQisJ?usp=sharing).
2. Place I3D_features and TEM_scores into the folder `data`.

## 4 Environment

- Hardware: GPU
- Framework:
  - PaddlePaddle >= 2.0.0

## 5 Quick start

### step1: clone

gi t clone git@github.com:rainyBJ/RTD_RePro.git

### step2: prepare dataset

1. according to 3
2. Use **dataset_converter.py** converting it to the paddle form

### step3:download chkpt

1. Download **checkpoint_best_sum_ar.pth** from [link](https://drive.google.com/file/d/1h20GnPhaJP3QkwVspn_ndXevJ97FGpE6/view?usp=sharing)

2. Use **chkpt_converter.py** converting it to the paddle form **checkpoint_best_sum_ar.pdparams**

### step4:test

```bash
python -m main --window_size 100 --batch_size 32 --lr 1e-4 --stage 3 --epochs 20 --num_queries 32 --point_prob_normalize --eval --resume checkpoint_best_sum_ar.pdparams 
```



