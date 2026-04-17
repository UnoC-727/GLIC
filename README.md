# CVPR2026-Adaptive Learned Image Compression with Graph Neural Networks



Our code and checkpoints for this repository are still being cleaned up, and will be released as soon as possible. In the meantime, if you are interested, please feel free to check out our previous paper and open-source repository: [Content-Aware Mamba for Learned Image Compression (ICLR 2026)](https://openreview.net/forum?id=WwDNiisZQm), [CMIC GitHub](https://github.com/UnoC-727/CMIC).




## Introduction

This repository will present the offical [Pytorch](https://pytorch.org/) implementation of [Adaptive Learned Image Compression with Graph Neural Networks (CVPR2026)](https://arxiv.org/abs/2603.25316). 

**Abstract:**
Efficient image compression relies on modeling both local and global redundancy. Most state-of-the-art (SOTA) learned image compression (LIC) methods are based on CNNs or Transformers, which are inherently rigid. Standard CNN kernels and window-based attention mechanisms impose fixed receptive fields and static connectivity patterns, which potentially couple non-redundant pixels simply due to their proximity in Euclidean space. This rigidity limits the model's ability to adaptively capture spatially varying redundancy across the image, particularly at the global level. To overcome these limitations, we propose a content-adaptive image compression framework based on Graph Neural Networks (GNNs). Specifically, our approach constructs dual-scale graphs that enable flexible, data-driven receptive fields. Furthermore, we introduce adaptive connectivity by dynamically adjusting the number of neighbors for each node based on local content complexity. These innovations empower our Graph-based Learned Image Compression (GLIC) model to effectively model diverse redundancy patterns across images, leading to more efficient and adaptive compression. Experiments demonstrate that GLIC achieves state-of-the-art performance, achieving BD-rate reductions of 19.29%, 21.69%, and 18.71% relative to VTM-9.1 on Kodak, Tecnick, and CLIC, respectively.



