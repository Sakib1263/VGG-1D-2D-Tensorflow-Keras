# VGG-Model-Builder-Tensorflow-Keras

This repository contains an One-Dimentional (1D) and Two-Dimentional (2D) versions of original variants of VGG developed in KERAS along with implementation guidance (DEMO) in Jupyter Notebook. Read more about VGG Models in this original paper [1].  
The models in this repository have been built following the original paper's implementation as much as possible, though more efficient implementation could be possible due to the advancements in this field since then. On the contrary, the models contain BatchNormalization (BN) blocks after Convolutional blocks and before activation, which is deviant from the original implementation. Read more about BN in this paper [2] 
The models implemented in this repository are: [1]
1. VGG11 (A)
2. VGG13 (B)
3. VGG16 (C)
4. VGG16_v2 (D)
5. VGG19 (E)

## VGG Architectures
Details of the VGG models developed are provided in the following table from the original paper.
![VGG Architecture Params](https://github.com/Sakib1263/VGG-Model-Builder-KERAS/blob/main/Documents/Images/VGG.png "VGG Parameters")  

## Supported Features
The speciality about this model is its flexibility. The user has the option for: 
1. Choosing any of 4 available VGG models for either 1D or 2D tasks.
2. Number of input kernel/filter, commonly known as Width of the model.
3. Number of classes for Classification tasks and number of extracted features for Regression tasks.
4. Number of Channels in the Input Dataset.
Details of the process are available in the DEMO provided in the codes section. The datasets used in the DEMO as also available in the 'Documents' folder.  

A 3D view of the VGG Architecture is provided below for better clarification:
![VGG Architecture](https://github.com/Sakib1263/VGG-Model-Builder-KERAS/blob/main/Documents/Images/VGG%20Model.png "VGG Architecture 3D")  

## References  
**[1]** Simonyan, K., & Zisserman, A. (2021). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv.org. Retrieved 30 August 2021, from https://arxiv.org/abs/1409.1556.  
**[2]** Ioffe, S., & Szegedy, C. (2021). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. arXiv.org. Retrieved 30 August 2021, from https://arxiv.org/abs/1502.03167.  
