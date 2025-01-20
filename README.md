
# Supervised and Semi-Supervised 3D Cardiac Segmentation
by Huy T. Nguyen
## Introduction
Official code for Professor Stephanie Bricq's project. This repo also can be used for later publication purpose.
## Requirements
This repository is based on PyTorch 1.8.0, CUDA 11.1 and Python 3.6.13. All experiments in our paper were conducted on NVIDIA GeForce RTX 3090 GPU with an identical experimental setting.
## Usage
I have provided `code`, `data_split` and `models` for LA and ACDC dataset.

To train a supervised model,
```
python ./code/LA_supervised_train.py  #for LA training
python ./code/ACDC_supervised_train.py  #for ACDC training
```
To train a semi-supervised model,
```
python ./code/LA_MT_train.py  #for LA training
python ./code/ACDC_MT_train.py  #for ACDC training
``` 

To test a model,
```
python ./code/test_LA.py  #for LA testing
python ./code/test_ACDC.py  #for ACDC testing
```

## Acknowledgements
Our code is largely based on [SS-Net](https://github.com/ycwu1997/SS-Net). Thanks for these authors for their valuable work, hope our work can also contribute to related research.

## Questions
If you have any questions, welcome contact me at 'thanh-huy_nguyen@etu.u-bourgogne.fr' or 'bihuyz@gmail.com'



