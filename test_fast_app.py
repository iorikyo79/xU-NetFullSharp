import pytest
from fastapi.testclient import TestClient
from PIL import Image
from io import BytesIO
import json
import os
import sys

# Add project root to sys.path to allow importing fast_app
# This might be needed if tests are run from a different directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))) # if test_fast_app.py is in root
# If test_fast_app.py is in a 'tests' subdirectory, it would be:
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from fast_app import app
except ImportError as e:
    print(f"Error importing fast_app: {e}. Ensure fast_app.py is in the Python path.")
    # As a fallback for environments where sys.path manipulation in code isn't effective immediately
    # or if the structure is different, we might need to rely on PYTHONPATH or pytest path configurations.
    # For now, we'll proceed assuming the import will work in the execution environment.
    # A placeholder app to allow tests to be defined, though they will fail if import truly fails.
    from fastapi import FastAPI
    app = FastAPI()


client = TestClient(app)

# --- Helper Functions & Fixtures ---

def create_dummy_image_bytes(size=(100, 100), color="blue", format="PNG") -> BytesIO:
    """Creates a dummy image in memory and returns its bytes."""
    img = Image.new("RGB", size, color)
    byte_io = BytesIO()
    img.save(byte_io, format)
    byte_io.seek(0)
    return byte_io

@pytest.fixture(scope="module")
def weights_mapping():
    """Loads weights_mapping.json for use in tests."""
    try:
        with open("weights_mapping.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        pytest.fail("weights_mapping.json not found. This file is required for tests.")
        return {} # Should not be reached if pytest.fail works

# --- Tests for /openapi.json endpoint ---

def test_read_openapi_spec():
    """Test retrieving the OpenAPI schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    content = response.json()
    assert "openapi" in content
    assert "info" in content
    assert "paths" in content
    # Check if the server URL was updated (assuming previous subtask was to change it to 8000)
    assert any(server["url"] == "http://localhost:8000" for server in content.get("servers", [])), \
        "Server URL was not updated to http://localhost:8000 in openapi.json"


# --- Tests for /models endpoint ---

def test_get_models(weights_mapping):
    """Test retrieving the list of available models."""
    response = client.get("/models")
    assert response.status_code == 200
    content = response.json()
    assert "models" in content
    models_from_api = content["models"]
    assert isinstance(models_from_api, list)
    assert len(models_from_api) > 0, "Model list should not be empty"

    api_model_names = set()
    for model_info in models_from_api:
        assert "name" in model_info
        assert "description" in model_info
        assert isinstance(model_info["name"], str)
        assert isinstance(model_info["description"], str)
        api_model_names.add(model_info["name"])

    # Verify that all models listed in the API are present in weights_mapping.json
    # (fast_app.py filters models based on WEIGHTS_MAPPING)
    for name_from_api in api_model_names:
        assert name_from_api in weights_mapping, \
            f"Model '{name_from_api}' from API not found in weights_mapping.json"

    # Also verify that all models in weights_mapping are exposed by the API
    # (unless there's a reason for some to be hidden, which is not the case here)
    for name_from_weights_map in weights_mapping.keys():
        assert name_from_weights_map in api_model_names, \
            f"Model '{name_from_weights_map}' from weights_mapping.json not found in API response"


# --- Tests for /process_image endpoint ---

def test_process_image_success(weights_mapping):
    """Test successful image processing with a valid model and image."""
    if not weights_mapping:
        pytest.skip("Skipping test_process_image_success as weights_mapping is empty or failed to load.")

    # Use the first model from weights_mapping for the test
    # Ensure a model that has its weights file actually present is chosen.
    # For this test, we assume 'UNET' is a safe bet for having weights.
    # A more robust way would be to check for os.path.exists(weights_path) for each model.
    model_name = "UNET" # Or list(weights_mapping.keys())[0]
    if model_name not in weights_mapping:
         pytest.fail(f"Test model '{model_name}' not found in weights_mapping.json.")

    weights_path = weights_mapping[model_name]
    # This test requires the actual weight file to be present where fast_app.py expects it.
    # If not, fast_app.py's load_weights will raise an error.
    # We'll proceed assuming the environment is set up with accessible weights for 'UNET'.
    if not os.path.exists(weights_path):
         pytest.warning(f"Weight file for model {model_name} not found at {weights_path}. Test might fail if model loading is strict.")


    dummy_image = create_dummy_image_bytes()
    files = {"image": ("dummy.png", dummy_image, "image/png")}
    data = {"model_name": model_name}

    response = client.post("/process_image", files=files, data=data)

    assert response.status_code == 200
    content = response.json()
    assert content.get("success") is True, f"API call failed with: {content.get('error')}"
    assert "original_image" in content and content["original_image"]
    assert "prediction_image" in content and content["prediction_image"]
    assert "bone_only_image" in content and content["bone_only_image"]
    assert content.get("model_name") == model_name
    assert "original_size" in content and len(content["original_size"]) == 2


def test_process_image_no_image():
    """Test processing request without an image file."""
    data = {"model_name": "UNET"} # Any valid model name
    # The 'files' parameter is omitted or an empty dict is sent
    response = client.post("/process_image", data=data) # No files part
    # FastAPI should return 422 if 'image: UploadFile = File(...)' is a required field
    assert response.status_code == 422
    content = response.json()
    assert "detail" in content # FastAPI's default error response for validation errors


def test_process_image_invalid_model_name():
    """Test processing with an invalid model name."""
    dummy_image = create_dummy_image_bytes()
    files = {"image": ("dummy.png", dummy_image, "image/png")}
    invalid_model_name = "THIS_MODEL_DOES_NOT_EXIST_12345"
    data = {"model_name": invalid_model_name}

    response = client.post("/process_image", files=files, data=data)

    # Expecting 400 or a specific error code defined in fast_app.py for invalid model
    # fast_app.py returns a JSON with "success": False and an "error" field.
    assert response.status_code == 200 # The endpoint itself returns 200 but with success=False
    content = response.json()
    assert content.get("success") is False
    assert "error" in content
    assert invalid_model_name in content["error"] or "not available" in content["error"]


def test_process_image_invalid_file_type():
    """Test processing with a non-image file type."""
    # Create dummy text file bytes
    text_content = b"This is not an image file."
    dummy_text_file = BytesIO(text_content)

    files = {"image": ("dummy.txt", dummy_text_file, "text/plain")}
    data = {"model_name": "UNET"} # Any valid model name

    response = client.post("/process_image", files=files, data=data)

    # fast_app.py checks content_type and returns success=False if not image
    assert response.status_code == 200 # Endpoint returns 200, but success=False
    content = response.json()
    assert content.get("success") is False
    assert "error" in content
    assert "Uploaded file is not an image" in content["error"]

# To run these tests, you would typically use `pytest` in the terminal.
# Ensure `fast_app.py` and `weights_mapping.json` are in the same directory as this test file,
# or adjust paths accordingly (e.g., using a 'tests' subdirectory and modifying sys.path).
# Also, `python-multipart` needs to be installed for `TestClient` to correctly send `files`.
# The `tensorflow` and other model dependencies also need to be available.
# A placeholder for model weight files might be needed if actual files are large/unavailable in CI.
# For `test_process_image_success`, actual weight files are needed for the model `UNET`.
# If `fast_app.py` is in the root and this test file is also in the root, the `sys.path` modification
# `sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))` is correct.
# If this test file is in a `tests/` subdirectory, then it should be `..` to find `fast_app`.
# Assuming tests are in the root for now.
# (Adjusted sys.path.append('.') to sys.path.insert(0, ...) for robustness)
# (Added a try-except for the import of `app` for better error visibility if path issues occur)
# (Added check for server URL in openapi.json test)
# (Made weights_mapping a module-scoped fixture)
# (Skipped success test if weights_mapping is not loaded)
# (Added warning if weight file for test model is missing)
# (Adjusted assertion for invalid model name to expect 200 with success=False)
# (Adjusted assertion for invalid file type to expect 200 with success=False)
# (Corrected file missing error code to 422 for FastAPI)
# (Clarified model list verification in test_get_models)
