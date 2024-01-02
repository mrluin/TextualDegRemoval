# TextualDegRemoval
Implementation of [**Improving Image Restoration through Removing Degradations in Textual Representations**](https://arxiv.org/abs/2312.17334)

[![arXiv](https://img.shields.io/badge/arXiv-2312.17334-b10.svg)](https://arxiv.org/abs/2312.17334)

The main contributions of this paper:
* We introduce a new perspective for image restoration, i.e.,
performing restoration first in textual space where degradations
and content are loosely coupled, and then utilizing
the restored content to guide image restoration
* To address the cross-modal assistance, we propose to embed
an image-to-text mapper and textual restoration module
into CLIP-equipped text-to-image models to generate
clear guidance from degraded images.
* Extensive experiments on multiple tasks demonstrate that
our method improves the performance of state-of-the-art
image restoration networks

### 1. Abstract

In this paper, we introduce a new perspective for improving image restoration by removing degradation in the textual representations of a given degraded image. Intuitively, restoration is much easier on text modality than image one. For example, it can be easily conducted by removing degradation-related words while keeping the contentaware words. Hence, we combine the advantages of images in detail description and ones of text in degradation removal to perform restoration. To address the cross-modal assistance, we propose to map the degraded images into textual representations for removing the degradations, and then convert the restored textual representations into a guidance image for assisting image restoration. In particular, We ingeniously embed an image-to-text mapper and text restoration module into CLIP-equipped text-to-image models to generate the guidance. Then, we adopt a simple coarse-to-fine approach to dynamically inject multiscale information from guidance to image restoration networks. Extensive experiments are conducted on various image restoration tasks, including deblurring, dehazing, deraining, and denoising, and all-in-one image restoration. The results showcase that our method outperforms state-of-the-art ones across all these tasks

### 2. Motivation

<p align="center"><img src="assets/intro_motivation.png" width="95%"></p>

### 3. Framework

<p align="center"><img src="assets/main_framework.png" width="95%"></p>


code will be publicly released soon!