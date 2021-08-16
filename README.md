# VGG-Model-Builder-KERAS

This repository contains an One-Dimentional (1D) and Two-Dimentional (2D) versions of original variants of VGG developed in KERAS along with implementation guidance (DEMO) in Jupyter Notebook. Read more about VGG Models in this original paper: https://arxiv.org/abs/1409.1556v6.  
The models in this repository have been built following the original paper's implementation as much as possible, though more efficient implementation could be possible due to the advancements in this field since then. On the contrary, the models contain BatchNormalization (BN) blocks after Convolutional blocks and before activation, which is deviant from the original implementation. Read more about BN in this paper: https://arxiv.org/abs/1502.03167v3.  
The models implemented in this repository are:
1. VGG11 (VGG_A in the Original Paper)
2. VGG16 (VGG_D)
3. VGG16_v2 (VGG_C)
4. VGG19 (VGG_E)

# VGG Architectures
![VGG Architecture Params](https://github.com/Sakib1263/VGG-Model-Builder-KERAS/blob/main/Documents/Images/VGG.png "VGG Parameters")  

The speciality about this model is its flexibility. Apart from choosing any of 4 available VGG models for 1D or 2D tasks, one can easily change the parameters such as number of input kernels/filters, number of classes for Classification tasks and number of extracted features for Regression tasks, etc. Details of the process are available in the DEMO provided in the codes section.  

A 3D view of the VGG Architecture is provided below for better clarification:
![VGG Architecture](https://github.com/Sakib1263/VGG-Model-Builder-KERAS/blob/main/Documents/Images/VGG%20Model.png "VGG Architecture 3D")  
