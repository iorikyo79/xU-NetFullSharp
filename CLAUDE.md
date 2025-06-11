# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is xU-NetFullSharp, a deep learning framework for chest X-ray bone shadow suppression using various U-Net-based CNN architectures. The project implements multiple state-of-the-art models including the novel xU-NetFullSharp architecture.

## Environment Setup

Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate bone_suppression
```

## Core Commands

### Training Models
```bash
python train.py --model_name <MODEL_NAME> --data_path <DATASET_PATH>
python train.py --model_name <MODEL_NAME> --data_path <DATASET_PATH> --weights_path <WEIGHTS_PATH>
```

### Testing Models
```bash
# Internal testing (requires JSRT and BSE_JSRT subdirectories)
python test.py --model_name <MODEL_NAME> --test_variant internal --data_path <DATASET_PATH>

# External testing (processes all images in folder)
python test.py --model_name <MODEL_NAME> --test_variant external --data_path <DATASET_PATH>
```

### Running Web Interface
```bash
python app.py
```

## Supported Models

Available model names for training/testing:
- `"UNET"`, `"ATT_UNET"`, `"DEEP_RESUNET"` - U-Net variants
- `"UNETPP"`, `"ATT_UNETPP"` - U-Net++ variants  
- `"UNET3P"` - U-Net3+ architecture
- `"UNET_SHARP"` - U-Net# with redesigned skip connections
- `"XUNETFS"`, `"ATT_XUNETFS"` - Novel xU-NetFullSharp architectures (primary contribution)
- `"KALISZ_AE"` - Kalisz-Marczyk Autoencoder
- `"UNET_RES18"`, `"FPN_RES18"`, `"FPN_EF0"` - DeBoNet ensemble variants

## Architecture Structure

### Core Components
- `models/` - Contains all neural network architectures
- `utils.py` - Training utilities, evaluation metrics, data loading functions
- `generator.py` - Data generator classes and preprocessing functions
- `weights_mapping.json` - Maps model names to their best trained weights
- `train.py` - Main training script with model selection logic
- `test.py` - Testing/inference script for model evaluation
- `app.py` - Flask web interface for interactive testing

### Key Technical Details
- All models expect 512x512 grayscale input images
- Default batch size: 10 (training), 5 (inference)
- Models use TensorFlow/Keras with custom loss functions (MAE, MSE, PSNR, SSIM)
- Preprocessing includes random image inversion (50% probability during training)
- Pre-trained weights are automatically loaded via `weights_mapping.json`

### Data Structure Requirements
Training data should be organized as:
```
dataset/
├── train/
│   ├── JSRT/        # Original X-ray images
│   └── BSE_JSRT/    # Ground truth bone-suppressed images
└── val/
    ├── JSRT/
    └── BSE_JSRT/
```

External testing processes any images in the specified folder and saves outputs to `outputs/` directory.

## xU-NetFullSharp Architecture

The novel xU-NetFullSharp combines:
- U-Net# base architecture with redesigned skip connections
- Bidirectional multi-scale skip connections (from U-Net3+)
- xUnit activation functions replacing ReLU
- Dilated convolution blocks with growing dilation rates

This architecture achieved best performance in bone shadow suppression while preserving anatomical details.