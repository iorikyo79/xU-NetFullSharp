# xU-NetFullSharp REST API Documentation

This Flask application provides REST API endpoints for chest X-ray bone shadow suppression using various deep learning models.

## Base URL
```
http://localhost:5000
```

## Endpoints

### 1. API Documentation
**GET** `/docs`

Interactive Swagger UI documentation interface for exploring and testing the API.

**Response:** HTML page with Swagger UI

**Status Codes:**
- `200 OK` - Success

---

### 2. OpenAPI Specification
**GET** `/openapi.json`

Returns the OpenAPI 3.0.3 specification in JSON format.

**Response:**
```json
{
  "openapi": "3.0.3",
  "info": {
    "title": "xU-NetFullSharp Bone Suppression API",
    "version": "1.0.0"
  },
  "paths": {...}
}
```

**Status Codes:**
- `200 OK` - Success
- `404 Not Found` - OpenAPI specification file not found

---

### 3. Get Available Models
**GET** `/get_models`

Returns a list of all available bone suppression models with their descriptions.

**Response:**
```json
{
  "models": [
    {
      "name": "XUNETFS",
      "description": "xU-NetFullSharp - Novel architecture (Our best model)"
    },
    {
      "name": "UNET",
      "description": "Standard U-Net - Basic architecture for image segmentation"
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Success

---

### 4. Process Image
**POST** `/process_image`

Processes an uploaded chest X-ray image using the specified model to suppress bone shadows.

**Request Format:** `multipart/form-data`

**Parameters:**
- `image` (file, required) - Chest X-ray image file (PNG/JPG/JPEG)
- `model_name` (string, required) - Name of the model to use for processing

**Available Model Names:**
- `UNET` - Standard U-Net
- `ATT_UNET` - Attention U-Net  
- `DEEP_RESUNET` - Deep Residual U-Net
- `UNETPP` - U-Net++
- `ATT_UNETPP` - Attention U-Net++
- `UNET3P` - U-Net3+
- `UNET_SHARP` - U-Net#
- `XUNETFS` - xU-NetFullSharp (Recommended)
- `ATT_XUNETFS` - Attention xU-NetFullSharp
- `KALISZ_AE` - Kalisz-Marczyk Autoencoder
- `UNET_RES18` - U-Net ResNet-18
- `FPN_RES18` - FPN ResNet-18
- `FPN_EF0` - FPN EfficientNet-B0

**Success Response:**
```json
{
  "success": true,
  "model_name": "XUNETFS",
  "model_description": "xU-NetFullSharp - Novel architecture (Our best model)",
  "original_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "prediction_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "bone_only_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "original_size": [512, 512],
  "message": "Bone shadow suppression completed successfully!"
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "No image uploaded"
}
```

**Response Fields:**
- `success` (boolean) - Whether the operation was successful
- `model_name` (string) - Name of the model used
- `model_description` (string) - Description of the model
- `original_image` (string) - Base64 encoded original image
- `prediction_image` (string) - Base64 encoded bone-suppressed image
- `bone_only_image` (string) - Base64 encoded bone-only image (original - prediction)
- `original_size` (array) - Original image dimensions [width, height]
- `message` (string) - Success message
- `error` (string) - Error message (only in error responses)

**Status Codes:**
- `200 OK` - Success (even for processing errors, check `success` field)

**Error Cases:**
- No image uploaded
- No model selected
- Invalid model name
- Image processing errors
- Model loading errors

---

### 5. Web Interface
**GET** `/`

Serves the HTML web interface for interactive image processing.

**Response:** HTML page

**Status Codes:**
- `200 OK` - Success

## Image Processing Details

### Input Requirements
- **Format:** PNG, JPG, or JPEG
- **Color:** Automatically converted to grayscale
- **Size:** Any size (automatically resized to 512x512 for processing)

### Processing Pipeline
1. Image is converted to grayscale
2. Resized to 512x512 pixels for model input
3. Normalized to [0,1] range
4. Processed by the selected deep learning model
5. Output is resized back to original dimensions
6. Three images are returned:
   - Original image
   - Bone-suppressed image (prediction)
   - Bone-only image (difference between original and prediction)

### Output Format
All images are returned as base64-encoded PNG data URLs, ready for display in web browsers.

## Example Usage

### cURL Example
```bash
curl -X POST http://localhost:5000/process_image \
  -F "image=@chest_xray.png" \
  -F "model_name=XUNETFS"
```

### Python Example
```python
import requests

url = "http://localhost:5000/process_image"
files = {"image": open("chest_xray.png", "rb")}
data = {"model_name": "XUNETFS"}

response = requests.post(url, files=files, data=data)
result = response.json()

if result["success"]:
    print(f"Processing completed with {result['model_name']}")
    # result["prediction_image"] contains the bone-suppressed image
else:
    print(f"Error: {result['error']}")
```

## Model Performance

The `XUNETFS` (xU-NetFullSharp) model is recommended as it achieved the best performance in research evaluations:
- Best MAE: 0.0071
- Best MSE: 0.0003  
- Best SSIM: 0.9846
- Highest expert rating for bone shadow suppression quality

## Server Configuration

The Flask application runs on:
- **Host:** 0.0.0.0 (all interfaces)
- **Port:** 5000
- **Debug Mode:** Enabled

To start the server:
```bash
python app.py
```