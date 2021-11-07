## 1 Introduction

This project reproduces RTD based on paddlepaddle framework. RTD is a simple and end-to-end learnable framework (RTD-Net) for direct action proposal generation, by re-purposing a Transformer-alike architecture. Thanks to the parallel decoding of multiple proposals with explicit context modeling, our RTD-Net outperforms the previous state-of-the-art methods in temporal action proposal generation task on THUMOS14 and also yields a superior performance for action detection on this dataset. In addition, free of NMS post-processing, our detection pipeline is more efficient than previous methods.

### Paper

[RTD-Net (ICCV 2021)](https://arxiv.org/pdf/2102.01894.pdf)

 "Relaxed Transformer Decoders for Direct Action Proposal Generation", accepted in ICCV 2021.

### Reference project

https://github.com/MCG-NJU/RTD-Action

## 2 Accuracy

| Dataset                                 | AR@50 | AR@100 | AR@200 | AR@500 | checkpoint                                                   |
| --------------------------------------- | ----- | ------ | ------ | ------ | ------------------------------------------------------------ |
| THUMOS14 (eval every 1 epoch, PyTorch)  | 41.52 | 49.33  | 56.41  | 62.91  | [link](https://drive.google.com/file/d/1h20GnPhaJP3QkwVspn_ndXevJ97FGpE6/view?usp=sharing)(PyTorch) |
| THUMOS14 (eval every 2 epochs, PyTorch) | 40.66 | 48.58  | 55.21  | 61.90  | ./log/torch_best_sum_ar.txt                                  |
| THUMOS14 (eval every 2 epochs, Paddle)  | 40.04 | 48.15  | 54.79  | 61.55  | ./log/paddle_beast_sum_ar.txt                                |
| THUMOS14 (eval every 1 epoch, PyTorch)  | 40.13 | 48.73  | 55.98  | 62.09  | ./log/torch_best_sum_ar_eval1.txt                            |
| THUMOS14 (eval every 1 epoch, Paddle)   | 40.23 | 48.66  | 55.24  | 62.22  | ./log/paddle_beast_sum_ar_eval1.txt                          |

## 3 Dataset

To reproduce the results in THUMOS14 without further changes:

1. Download the data from [GoogleDrive](https://drive.google.com/drive/folders/13KwgSgeZKWwIYE77PVo4_dvZhf8qQisJ?usp=sharing).

## 4 Environment

- Hardware: GPU
- Framework:
  - PaddlePaddle >= **2.2.0.rc0 !!!(低版本会有问题)**

## 5 Quick start

### step1: clone

gi t clone git@github.com:rainyBJ/RTD_RePro.git

### step2: prepare dataset

1. according to 3
2. Use **dataset_converter.py** converting it to the paddle form
3. put them in ```./data_paddle/```

### step3:download chkpt

1. Download **checkpoint_best_sum_ar.pdparams**  & **checkpoint_initial.pdparams** from [link](https://aistudio.baidu.com/aistudio/datasetdetail/114662) (Paddle)
2. put them in the ```./```

### step4:test(from best chkpt)

```bash
sh val.sh 
```

### step5:train(from scratch)

``` bash
sh run.sh
```

## 6 AI Studio

### link

[AI Studio 项目](https://aistudio.baidu.com/aistudio/projectdetail/2559085?shared=1)（2.2.0rc0）

### dataset

unzip ~/data/data112050/归档.zip -d ~/work/data_paddle/

### chkpt

cp ~/data/data114662/* ~/work

### eval

sh val.sh

### train

sh train.sh

---

see logs in folder ./log/xxx.log
