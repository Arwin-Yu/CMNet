# CMNet(Improve Convolutional Networksconvolutio From Metaformer)

## 该项目主要是来自pytorch官方torchvision模块中的源码

* https://github.com/pytorch/vision/tree/main/torchvision/models/segmentation

## 环境配置：

* matplotlib==3.5.2
* numpy==1.22.3
* timm==0.5.4
* torch==1.11.0
* torchsummary==1.5.1
* torchvision==0.12.0
* tqdm==4.64
* python==3.6/3.7/3.8/3.9

## 文件结构：

```

  ├── CMNet: 模型的backbone 

  ├── exp: 论文中关于实验模型

  ├── train_exp.py: 论文中关于实验的训练代码 

  ├── train.py: CMNet的训练代码

  ├── preprocess.py: CMNet训练（验证）ImageNet1K时的数据预处理

  ├── train_engine.py: CMNet训练脚本中的一些子功能

 
```
 

## 数据集，本例程使用的是ImageNet2012数据集

 

## 训练方法

 

## 注意事项
 
## Pytorch官方实现的FCN网络框架图

![torch_fcn](torch_fcn.png)
