# Gemini Code Assistant Workspace

This document provides a summary of the project structure and key commands for the Gemini code assistant.

## Project Summary

This project implements a deep learning framework for bone shadow suppression in chest X-rays. The core of the project is a novel neural network architecture named xU-NetFullSharp, along with several other U-Net based models for comparison. The project includes scripts for training and testing these models on various datasets, as well as a Flask-based web application for interactive bone shadow suppression.

**Recent Updates (2025.01)**:
- Fixed EfficientNet compatibility issues with Keras 2.15+
- Enhanced uv package manager workflow
- Added comprehensive Python version compatibility (3.8-3.10)
- Improved web application interface
- Updated dependency management and troubleshooting documentation

## Key Files and Directories

- **`app.py`**: A web application for interacting with the models.
- **`train.py`**: Script for training the neural network models.
- **`test.py`**: Script for testing the trained models.
- **`generator.py`**: A data generator for feeding images to the models.
- **`utils.py`**: Utility functions used by other scripts.
- **`models/`**: Contains the Python implementations of the different neural network architectures.
- **`data/`**: Directory for storing training, validation, and test datasets.
- **`weights/`**: Contains the pre-trained weights for the models.
- **`outputs/`**: Default directory for saving the output of the `test.py` script.
- **`requirements.txt`**: Defines the Python dependencies for the project.
- **`weights_mapping.json`**: Maps model names to their corresponding weight files.

## Core Commands

### Environment Setup

#### ⚠️ Python Version Compatibility

**Important**: This project requires **Python 3.8-3.10** due to TensorFlow 2.10 compatibility limitations.

#### Option 1: Using uv (Recommended)

```bash
# Create virtual environment with Python 3.10
uv venv --python 3.10

# Install dependencies
uv pip install -r requirements.txt

# Use uv run for automatic environment management (no activation needed)
uv run python app.py
```

#### Option 2: Traditional Method

```bash
# Create and activate virtual environment
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Training a Model

To train a model, use the `train.py` script.

- **Basic training (using uv):**
  ```bash
  uv run python train.py --model_name <model_name> --data_path <path_to_dataset>
  ```
- **Training with pre-trained weights:**
  ```bash
  uv run python train.py --model_name <model_name> --data_path <path_to_dataset> --weights_path <path_to_weights>
  ```

### Testing a Model

To test a model, use the `test.py` script.

```bash
uv run python test.py --model_name <model_name> --test_variant <external|internal> --data_path <path_to_dataset>
```

### Running the Web Application

To start the Flask web application for interactive bone shadow suppression:

```bash
uv run python app.py
```

The web interface will be available at:
- Local: http://127.0.0.1:5000
- Network: http://[your-ip]:5000

## Supported Models

The following models are supported:

- `UNET`
- `ATT_UNET`
- `DEEP_RESUNET`
- `UNETPP`
- `ATT_UNETPP`
- `UNET3P`
- `UNET_SHARP`
- `XUNETFS`
- `ATT_XUNETFS`
- `KALISZ_AE`
- `UNET_RES18`
- `FPN_RES18`
- `FPN_EF0`

## Data Directory Structure

- **Training/Validation Data:** The data directory should contain `train` and `val` subdirectories. Each of these should have `JSRT` (original images) and `BSE_JSRT` (ground truth) subdirectories.
- **Internal Test Data:** The test data directory should contain `JSRT` and `BSE_JSRT` subdirectories.
- **External Test Data:** The test data directory can contain any number of images. The script will process all of them.

## Troubleshooting

### Common Issues and Solutions

#### EfficientNet Compatibility Error

If you encounter:
```
AttributeError: module 'keras.utils' has no attribute 'generic_utils'
```

**Solution**: Update EfficientNet to version 1.1.1+:
```bash
uv pip install "efficientnet>=1.1.1"
```

#### TensorFlow Installation Issues

- **Python Version**: Ensure you're using Python 3.8-3.10 (not 3.11+)
- **Check Python version**: `python --version`
- **Install specific Python version**: `uv python install 3.10`

#### Keras API Issues

If you encounter Keras API errors, ensure you're using compatible versions:
```bash
uv pip install tensorflow>=2.10,<2.16
uv pip install efficientnet>=1.1.1
```

## Recent Updates (2025.01)

- **Fixed EfficientNet compatibility**: Updated to version 1.1.1+ for Keras 2.15+ support
- **Enhanced uv workflow**: Added comprehensive uv package management
- **Python version compatibility**: Added explicit Python 3.8-3.10 requirements
- **Web application**: Added Flask-based interactive interface
- **Improved error handling**: Added detailed troubleshooting documentation
