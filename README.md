# Noise-Enhanced Graph Contrastive Learning for Multimodal Recommendation Systems

This repo provides the source code & data of our paper NEGCL: [Noise-Enhanced Graph Contrastive Learning for Multimodal Recommendation Systems](https://github.com/HubuKG/NEGCL) 

## Overview

The structure of our model is available for viewing in the following:
<p align="center">
   <img src="NEGCL.png" width="900">
</p>

### 1. Prerequisites

Python==3.8,

Pytorch==2.3.0,

Install all requirements with ``pip install -r requirements.txt``.


### 2. Download data

Put the Baby, Clothing, and Sports datasets and other required data into the folder ``NEGCL/data`` by downloading from this link [Google Drive](https://drive.google.com/drive/folders/1BxObpWApHbGx9jCQGc8z52cV3t9_NE0f?usp=sharing).

### 3. Training on a local server using PyCharm.

Run NEGCL by ``python main.py`` with the default dataset as Baby. Specific dataset selection can be modified in `main.py`.

### 4. Training on a local server using Git Bash.

Run NEGCL by ``train.\`` with the default dataset is Baby. Specific dataset selection can be modified in `train.py`.


### 5. Train on a cloud server.

Run ``!bash train.sh``

### 6. Modify specific parameters.

You may specify other parameters in CMD or config with `configs/model/*.yaml` and `configs/dataset/*.yaml`. 
