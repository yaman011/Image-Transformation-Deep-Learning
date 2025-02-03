# Image Style Transfer using Convolutional Neural Networks (CNNs)

## Overview
This project implements **Artistic Style Transfer** using Convolutional Neural Networks (CNNs). The goal is to transfer the artistic style of one image onto the content of another using deep learning techniques. The method is based on the original paper "A Neural Algorithm of Artistic Style" and utilizes **VGG19** as the backbone model for feature extraction.

## Features
- Implements **artistic style transfer** by separating style and content features.
- Uses **pre-trained CNN models (VGG19, ResNet50)** for feature extraction.
- Custom **VGG19 implementation** with improvements in training efficiency.
- Optimized **loss functions** for balancing style and content representation.
- Experiments with different **training datasets (CIFAR-100, Caltech101)**.
- Performance enhancements using **AdamW optimizer**, **ReduceLROnPlateau**, and **Mixed Precision Training**.

## Dataset
This project works with various datasets:
- **Caltech101**: Initially used but proved insufficient due to its small size.
- **CIFAR-100**: Adopted later for training the new VGG19-based architecture.
- **Custom Style & Content Images**: Used for style transfer demonstrations.

## Model Architecture
### **VGG19-Based Feature Extraction**
- The pre-trained **VGG19 network** extracts high-level features from both **content** and **style** images.
- **Content Loss**: Measures feature map differences between the content image and generated image.
- **Style Loss**: Computes the difference between Gram matrices of style and generated images.
- **Total Loss**: Weighted sum of style and content losses.

### **Custom VGG19 Implementation Enhancements**
- Used **adaptive average pooling** for feature size consistency.
- Improved weight initialization with **Kaiming Initialization**.
- Replaced Adam optimizer with **AdamW** for better weight decay handling.
- Introduced **learning rate scheduling** to enhance stability.

## Results Summary
- Pre-trained VGG19 achieved the best results.
- ResNet50 struggled due to residual connections.
- Caltech101's small dataset led to weak generalization.
- Training optimizations improved speed and quality.

## Repository Contents
### **Project Files**
- **`Image_Transformation_Deep_Learning_Style_Transfer.ipynb`** – Jupyter Notebook containing the **code implementation** for artistic style transfer using VGG19, including model architecture, training, and evaluation.
- **`Neural_Style_Transfer_Report.pdf`** – A **detailed report** documenting the project’s methodology, experiments, results, and analysis.

## References
- **A Neural Algorithm of Artistic Style** - Leon A. Gatys, Alexander S. Ecker, Matthias Bethge ([Paper](https://arxiv.org/abs/1508.06576))
- **Perceptual Losses for Real-Time Style Transfer** - Justin Johnson, Alexandre Alahi, Li Fei-Fei ([Paper](https://arxiv.org/abs/1603.08155))
- **Pytorch Documentation** - [AMP Training](https://pytorch.org/docs/stable/notes/amp_examples.html), [AdamW Optimizer](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
