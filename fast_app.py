import sys
sys.path.append('.')

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Union, Dict, Any
from PIL import Image
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO
import json
import os

# Model Imports from models directory
from models.UNet import UNet
from models.AttentionUNet import AttentionUNet
from models.DeepResUNet import DeepResUNet
from models.UNetPlusPlus import UNetPlusPlus
from models.AttentionUNetPlusPlus import AttentionUNetPlusPlus
from models.UNet3Plus import UNet3Plus
from models.UNetSharp import UNetSharp
from models.XUNetFullSharp import XUNetFullSharp
from models.AttentionXUNetFullSharp import AttentionXUNetFullSharp
from models.KaliszAE import KaliszAE
from models.ResNet18UNet import ResNet18UNet
from models.ResNet18FPN import ResNet18FPN
from models.EfficientNetB0FPN import EfficientNetB0FPN


# Constants
IMG_WIDTH = 512
IMG_HEIGHT = 512

# Model Descriptions and Weights Mapping
MODEL_DESCRIPTIONS = {
    "XUNETFS": "xU-NetFullSharp - Novel architecture (Our best model)",
    "UNET": "Standard U-Net - Basic architecture for image segmentation",
    "ATT_UNET": "Attention U-Net - UNet with attention gates",
    "DEEP_RESUNET": "Deep Residual U-Net - UNet with residual connections",
    "UNETPP": "U-Net++ - UNet with nested and dense skip connections",
    "ATT_UNETPP": "Attention U-Net++ - UNet++ with attention gates",
    "UNET3P": "U-Net3+ - UNet with full-scale skip connections",
    "UNET_SHARP": "U-Net# - UNet with sharpened filters",
    "ATT_XUNETFS": "Attention xU-NetFullSharp - xU-NetFullSharp with attention mechanism",
    "KALISZ_AE": "Kalisz Autoencoder - Convolutional autoencoder by Kalisz et al.",
    "UNET_RES18": "ResNet18 U-Net - U-Net with ResNet18 encoder",
    "FPN_RES18": "FPN ResNet18 - Feature Pyramid Network with ResNet18 backbone",
    "FPN_EF0": "FPN EfficientNetB0 - Feature Pyramid Network with EfficientNetB0 backbone"
}

try:
    with open('weights_mapping.json', 'r') as f:
        WEIGHTS_MAPPING = json.load(f)
except FileNotFoundError:
    print("Error: weights_mapping.json not found. Please ensure the file exists in the project root.")
    WEIGHTS_MAPPING = {} # Fallback to empty dict if file not found

# Helper Functions
def get_weights_path(model_name: str) -> str:
    return WEIGHTS_MAPPING.get(model_name)

def load_weights(model: tf.keras.Model, weights_path: str):
    """Loads pre-trained weights into a Keras model.

    Args:
        model: The Keras model.
        weights_path: Path to the HDF5 weights file.
    """
    try:
        model.load_weights(weights_path)
        print(f"Successfully loaded weights from {weights_path} for model {model.name}")
    except Exception as e:
        print(f"Error loading weights for model {model.name} from {weights_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model weights: {e}")


def create_model(model_name: str, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)) -> tf.keras.Model:
    """Creates and returns the specified model."""
    models_zoo = {
        "UNET": UNet,
        "ATT_UNET": AttentionUNet,
        "DEEP_RESUNET": DeepResUNet,
        "UNETPP": UNetPlusPlus,
        "ATT_UNETPP": AttentionUNetPlusPlus,
        "UNET3P": UNet3Plus,
        "UNET_SHARP": UNetSharp,
        "XUNETFS": XUNetFullSharp,
        "ATT_XUNETFS": AttentionXUNetFullSharp,
        "KALISZ_AE": KaliszAE,
        "UNET_RES18": ResNet18UNet,
        "FPN_RES18": ResNet18FPN,
        "FPN_EF0": EfficientNetB0FPN,
    }
    model_class = models_zoo.get(model_name)
    if not model_class:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not found.")

    # For models that require specific parameters like num_classes or filters
    if model_name in ["UNET", "ATT_UNET", "DEEP_RESUNET", "UNETPP", "ATT_UNETPP", "UNET_SHARP", "XUNETFS", "ATT_XUNETFS", "KALISZ_AE"]:
        # These models typically take input_shape directly or imply num_classes=1 for segmentation
        model_instance = model_class(input_shape=input_shape)
    elif model_name == "UNET3P":
        model_instance = model_class(num_classes=1, input_shape=input_shape, deep_supervision=False)
    elif model_name in ["UNET_RES18", "FPN_RES18", "FPN_EF0"]:
        # These might have a different constructor signature, adjust as necessary
        # Assuming they can take input_shape and default to 1 class for segmentation
        model_instance = model_class(input_shape=input_shape, num_classes=1)
    else:
        # Default instantiation if no special parameters are needed
        model_instance = model_class(input_shape=input_shape)

    return model_instance

# Pydantic Models
class ModelInfo(BaseModel):
    name: str
    description: str

class ModelListResponse(BaseModel):
    models: List[ModelInfo]

class ProcessingSuccessResponse(BaseModel):
    success: bool = True
    model_name: str
    model_description: str
    original_image: str  # base64 encoded
    prediction_image: str  # base64 encoded
    bone_only_image: str  # base64 encoded
    original_size: List[int]  # [width, height]
    message: str

class ProcessingErrorResponse(BaseModel):
    success: bool = False
    error: str

ImageProcessingResponse = Union[ProcessingSuccessResponse, ProcessingErrorResponse]

# FastAPI App Instance
app = FastAPI(
    title="xU-NetFullSharp Bone Suppression API",
    description="REST API for chest X-ray bone shadow suppression using deep learning models",
    version="1.0.0"
)

# API Endpoints
@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_schema():
    try:
        with open("openapi.json", "r") as f:
            return JSONResponse(content=json.load(f))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="openapi.json not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error decoding openapi.json")


@app.get("/models", response_model=ModelListResponse)
async def get_models_list():
    """Returns a list of all available bone suppression models with their descriptions."""
    models_info = []
    for name, description in MODEL_DESCRIPTIONS.items():
        if name in WEIGHTS_MAPPING: # Only list models that have weights configured
            models_info.append(ModelInfo(name=name, description=description))
    return ModelListResponse(models=models_info)

@app.post("/process_image", response_model=ImageProcessingResponse)
async def process_image_endpoint(
    image: UploadFile = File(...),
    model_name: str = Form(...)
):
    """Processes an uploaded chest X-ray image using the specified model to suppress bone shadows."""
    if not image.content_type.startswith("image/"):
        return ProcessingErrorResponse(error="Uploaded file is not an image.")

    if model_name not in MODEL_DESCRIPTIONS or model_name not in WEIGHTS_MAPPING:
        return ProcessingErrorResponse(error=f"Model '{model_name}' is not available.")

    try:
        # Load image
        img_bytes = await image.read()
        pil_image = Image.open(BytesIO(img_bytes)).convert('L')
        original_size = [pil_image.width, pil_image.height]

        # Preprocess image
        img_resized = pil_image.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(np.expand_dims(img_array, axis=0), axis=-1) # (1, H, W, 1)

        # Create model and load weights
        model = create_model(model_name, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1))
        weights_path = get_weights_path(model_name)
        if not weights_path or not os.path.exists(weights_path):
             return ProcessingErrorResponse(error=f"Weights not found for model '{model_name}'. Path: {weights_path}")
        load_weights(model, weights_path)

        # Prediction
        prediction_batch = model.predict(img_batch)
        prediction_array = prediction_batch[0, ..., 0] # Remove batch and channel dims

        # Postprocess prediction
        # Resize prediction back to original image size
        prediction_pil = Image.fromarray((prediction_array * 255).astype(np.uint8))
        prediction_resized_pil = prediction_pil.resize((original_size[0], original_size[1]), Image.LANCZOS)

        # Ensure original image is also available in the correct format for bone-only calculation
        original_array_resized_for_calc = np.array(pil_image.resize((original_size[0], original_size[1]), Image.LANCZOS), dtype=np.float32) / 255.0
        prediction_for_calc = np.array(prediction_resized_pil, dtype=np.float32) / 255.0

        # Generate bone-only image (original - prediction)
        # Clamp values to [0, 1] before scaling to [0, 255]
        bone_only_array = np.clip(original_array_resized_for_calc - prediction_for_calc, 0, 1)
        bone_only_pil = Image.fromarray((bone_only_array * 255).astype(np.uint8))

        # Convert images to base64 PNG strings
        def img_to_base64_str(img_pil):
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

        original_b64 = img_to_base64_str(pil_image) # Send original uploaded image
        prediction_b64 = img_to_base64_str(prediction_resized_pil)
        bone_only_b64 = img_to_base64_str(bone_only_pil)

        return ProcessingSuccessResponse(
            model_name=model_name,
            model_description=MODEL_DESCRIPTIONS[model_name],
            original_image=f"data:image/png;base64,{original_b64}",
            prediction_image=f"data:image/png;base64,{prediction_b64}",
            bone_only_image=f"data:image/png;base64,{bone_only_b64}",
            original_size=original_size,
            message="Bone shadow suppression completed successfully!"
        )

    except HTTPException as http_exc: # Handles errors from create_model or load_weights
        return ProcessingErrorResponse(error=str(http_exc.detail))
    except Exception as e:
        print(f"Error processing image: {e}")
        # import traceback
        # traceback.print_exc()
        return ProcessingErrorResponse(error=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Note: The openapi.json served by FastAPI will be dynamically generated.
    # The static openapi.json file is for reference or alternative serving.
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Ensure all model classes used in create_model are defined or imported:
# UNet, AttentionUNet, DeepResUNet, UNetPlusPlus, AttentionUNetPlusPlus,
# UNet3Plus, UNetSharp, XUNetFullSharp, AttentionXUNetFullSharp, KaliszAE,
# ResNet18UNet, ResNet18FPN, EfficientNetB0FPN
# All seem to be imported.
