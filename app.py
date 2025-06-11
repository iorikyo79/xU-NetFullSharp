from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import sys
import os
import json
import glob
from io import BytesIO
import base64

# Add the xU-NetFullSharp directory to the Python path
#sys.path.append('xU-NetFullSharp')

# Import utilities and models from the xU-NetFullSharp project
from utils import load_weights, compile_model
from models.UNet3P import UNet3P
from models.UNetPP import UNetPP
from models.UNet import UNet
from models.Att_UNet import Att_UNet
from models.Att_UNetPP import Att_UNetPP
from models.DeepResUNet import DeepResUNet
from models.UNetSharp import UNetSharp
from models.DeBoNet import DeBoNet
from models.Kalisz_AE import KaliszAE
from models.xUNetFS import xUNetFS
from models.Att_xUNetFS import Att_xUNetFS

# Image size (models expect 512x512)
IMG_WIDTH = 512
IMG_HEIGHT = 512

# Load the weights mapping
#with open("xU-NetFullSharp/weights_mapping.json", "r") as file:
with open("weights_mapping.json", "r") as file:
    weights_mapping = json.load(file)

# Model descriptions for better user understanding
MODEL_DESCRIPTIONS = {
    "UNET": "Standard U-Net - Basic architecture for image segmentation",
    "ATT_UNET": "Attention U-Net - U-Net with attention mechanisms",
    "DEEP_RESUNET": "Deep Residual U-Net - U-Net with residual connections",
    "UNETPP": "U-Net++ - Nested U-Net architecture",
    "ATT_UNETPP": "Attention U-Net++ - U-Net++ with attention mechanisms",
    "UNET3P": "U-Net3+ - Full-scale connected U-Net",
    "UNET_SHARP": "U-Net# - U-Net with redesigned skip connections",
    "XUNETFS": "xU-NetFullSharp - Novel architecture (Our best model)",
    "ATT_XUNETFS": "Attention xU-NetFullSharp - xU-NetFullSharp with attention",
    "KALISZ_AE": "Kalisz-Marczyk Autoencoder - Alternative architecture",
    "UNET_RES18": "U-Net ResNet-18 - U-Net with ResNet backbone",
    "FPN_RES18": "FPN ResNet-18 - Feature Pyramid Network with ResNet",
    "FPN_EF0": "FPN EfficientNet-B0 - Feature Pyramid Network with EfficientNet"
}

# Function to get weights path by model name
def get_weights_path(model_name):
    try:
        return f"{weights_mapping[model_name]}"
    except KeyError:
        raise ValueError(f"Model {model_name} is not recognized.\nAvailable models: {list(weights_mapping.keys())}")

# Function to create model by name
def create_model(model_name):
    if model_name == "KALISZ_AE":
        return KaliszAE()
    elif model_name == "UNET3P":
        return UNet3P()
    elif model_name == "UNETPP":
        return UNetPP()
    elif model_name == "UNET":
        return UNet()
    elif model_name == "ATT_UNET":
        return Att_UNet()
    elif model_name == "ATT_UNETPP":
        return Att_UNetPP()
    elif model_name == "DEEP_RESUNET":
        return DeepResUNet()
    elif model_name == "UNET_SHARP":
        return UNetSharp()
    elif model_name == "UNET_RES18":
        return DeBoNet(COMPILE=False, NAME=model_name)
    elif model_name == "FPN_RES18":
        return DeBoNet(COMPILE=False, NAME=model_name)
    elif model_name == "FPN_EF0":
        return DeBoNet(COMPILE=False, NAME=model_name)
    elif model_name == "XUNETFS":
        return xUNetFS()
    elif model_name == "ATT_XUNETFS":
        return Att_xUNetFS()
    else:
        raise ValueError(f"Model {model_name} is not recognized.\nAvailable models: {list(weights_mapping.keys())}")

app = Flask(__name__, static_folder='.', static_url_path='', template_folder='.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/openapi.json', methods=['GET'])
def get_openapi_spec():
    """Return OpenAPI specification"""
    try:
        with open('openapi.json', 'r') as f:
            spec = json.load(f)
        return jsonify(spec)
    except FileNotFoundError:
        return jsonify({'error': 'OpenAPI specification not found'}), 404

@app.route('/docs')
def api_docs():
    """Serve Swagger UI for API documentation"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>xU-NetFullSharp API Documentation</title>
        <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
        <style>
            html {
                box-sizing: border-box;
                overflow: -moz-scrollbars-vertical;
                overflow-y: scroll;
            }
            *, *:before, *:after {
                box-sizing: inherit;
            }
            body {
                margin:0;
                background: #fafafa;
            }
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
        <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-standalone-preset.js"></script>
        <script>
            window.onload = function() {
                const ui = SwaggerUIBundle({
                    url: '/openapi.json',
                    dom_id: '#swagger-ui',
                    deepLinking: true,
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIStandalonePreset
                    ],
                    plugins: [
                        SwaggerUIBundle.plugins.DownloadUrl
                    ],
                    layout: "StandaloneLayout"
                });
            };
        </script>
    </body>
    </html>
    '''

@app.route('/get_models', methods=['GET'])
def get_models():
    """Return available models with descriptions"""
    models = []
    for model_name in weights_mapping.keys():
        models.append({
            'name': model_name,
            'description': MODEL_DESCRIPTIONS.get(model_name, "No description available")
        })
    return jsonify({'models': models})

@app.route('/process_image', methods=['POST'])
def process_image():
    """Process uploaded image and return inference results"""
    try:
        # Check if image and model are provided
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'})
        
        if 'model_name' not in request.form:
            return jsonify({'success': False, 'error': 'No model selected'})

        image_file = request.files['image']
        model_name = request.form['model_name']

        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})

        # Validate model name
        if model_name not in weights_mapping:
            return jsonify({'success': False, 'error': f'Invalid model: {model_name}'})

        # Process image
        image = Image.open(image_file).convert('L')  # Convert to grayscale
        original_size = image.size
        
        # Convert original image to base64 for display
        original_buffer = BytesIO()
        image.save(original_buffer, format="PNG")
        original_base64 = base64.b64encode(original_buffer.getvalue()).decode()

        # Resize for model input
        image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
        image_np = np.array(image_resized) / 255.0
        image_np = np.expand_dims(image_np, axis=0)
        image_np = np.expand_dims(image_np, axis=-1)  # Add channel dimension

        # Create and load model
        model = create_model(model_name)
        weights_path = get_weights_path(model_name)
        load_weights(model, weights_path)

        # Perform inference
        prediction = model.predict(image_np)
        prediction_np = np.squeeze(prediction)  # Remove batch and channel dimensions
        
        # Resize prediction back to original size
        prediction_image = Image.fromarray((prediction_np * 255).astype(np.uint8))
        prediction_image = prediction_image.resize(original_size)

        # Convert prediction to base64
        prediction_buffer = BytesIO()
        prediction_image.save(prediction_buffer, format="PNG")
        prediction_base64 = base64.b64encode(prediction_buffer.getvalue()).decode()

        # Create bone-only image (original - prediction)
        original_np = np.array(image) / 255.0
        prediction_resized_np = np.array(prediction_image) / 255.0
        bone_only_np = np.clip(original_np - prediction_resized_np, 0, 1)
        bone_only_image = Image.fromarray((bone_only_np * 255).astype(np.uint8))
        
        bone_only_buffer = BytesIO()
        bone_only_image.save(bone_only_buffer, format="PNG")
        bone_only_base64 = base64.b64encode(bone_only_buffer.getvalue()).decode()

        result = {
            'success': True,
            'model_name': model_name,
            'model_description': MODEL_DESCRIPTIONS.get(model_name, ""),
            'original_image': original_base64,
            'prediction_image': prediction_base64,
            'bone_only_image': bone_only_base64,
            'original_size': original_size,
            'message': 'Bone shadow suppression completed successfully!'
        }

    except Exception as e:
        result = {
            'success': False,
            'error': f'Processing failed: {str(e)}'
        }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)