# Selective Refinement Network for High Performance Face Detection

By [Cheng Chi](https://chicheng123.github.io/) and [Shifeng Zhang](http://www.cbsr.ia.ac.cn/users/sfzhang/)

### Introduction
This paper is accepted by AAAI 2019.

SRN is a real-time face detector, which performs superiorly on various scales of faces with a single deep neural network, especially for small faces. 

For more details, please refer to our [paper](https://arxiv.org/abs/1809.02693).


### Contents
1. [Requirements](#requirements)
2. [Preparation](#preparation)
3. [Evaluation](#evaluation)

### Requirements
- Torch == 0.3.1
- Torchvision == 0.2.1
- Python == 3.6
- CUDA CuDNN
- Numpy
- OpenCV

### Preparation
1. Clone the github repository. We will call the directory `$SRN_ROOT`
  ```Shell
  git clone https://github.com/ChiCheng123/SRN
  cd $SRN_ROOT
  ```

2. Compile extensions.
  ```Shell
  cd srn/extensions
  bash build_ext.sh
  ```

3. Download our trained model from [GoogleDrive](https://drive.google.com/open?id=1T4Qt99SdM7c8G4ZuC1igensY0bZdEETF) or [BaiduYun](https://pan.baidu.com/s/1ambmu1Bu6Oi7zTcEnigFyg) with extraction code `6fba`, and put it into the folder `$SRN_ROOT/model`.

4. Download [WIDER FACE](http://shuoyang1213.me/WIDERFACE/index.html) dataset, and link the image path with the project.
  ```Shell
  ln -sf $WIDER_FACE/images $SRN_ROOT/data/images
  ```

### Evaluation
Evaluate our model on WIDER FACE. We also integrate the [eval tool](http://shuoyang1213.me/WIDERFACE/index.html) of WIDER FACE. You can evaluate our model and get the final result with only one shell script. 
  ```Shell
  cd $SRN_ROOT/tools
  sh val.sh
  ```

If the max memory capacity of your GPU is 11G (1080TI) or 12G (TITANXP), please set the `max_size` in `val.sh` to 1400. You will get the result: `Easy: 96.5, Medium: 95.2, Hard: 89.6`.

If the max memory capacity of your GPU is 24G or larger, please set the `max_size` in `val.sh` to 2100. You will get the result: `Easy: 96.5, Medium: 95.3, Hard: 90.2`.

### To Do List
- [ ] Release the FP16 models to test images with size 2100*2100 on common GPUs
- [ ] Release the models with several backbones, i.e., ResNet-101, ResNet-152 and ResNet-18.
- [ ] Release the training codes


### Citation
If you find SRN useful in your research, please consider citing: 
```
@article{chi2018selective,
  title={Selective refinement network for high performance face detection},
  author={Chi, Cheng and Zhang, Shifeng and Xing, Junliang and Lei, Zhen and Li, Stan Z and Zou, Xudong},
  journal={arXiv preprint arXiv:1809.02693},
  year={2018}
}
```