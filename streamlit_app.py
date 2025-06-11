import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64

# Configuration
FASTAPI_URL = "http://localhost:8000"  # URL of your FastAPI backend

# Page Setup
st.set_page_config(page_title="X-ray Bone Suppression", layout="wide")
st.title("Chest X-ray Bone Suppression Web App")

# --- Helper Functions ---
def get_models():
    """Fetches the list of available models from the FastAPI backend."""
    try:
        response = requests.get(f"{FASTAPI_URL}/models")
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json().get("models", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching models: {e}")
        return []
    except json.JSONDecodeError:
        st.error("Error decoding model list from server.")
        return []

def base64_to_image(base64_string):
    """Converts a base64 encoded image string (with data URL prefix) to a PIL Image."""
    try:
        # Remove the "data:image/png;base64," prefix if it exists
        if "," in base64_string:
            base64_string = base64_string.split(',', 1)[1]
        img_bytes = base64.b64decode(base64_string)
        return Image.open(BytesIO(img_bytes))
    except Exception as e:
        st.error(f"Error decoding image: {e}")
        return None

# --- Sidebar for Model Selection ---
st.sidebar.header("⚙️ Controls")
models_list = get_models()
selected_model_name = None

if models_list:
    model_display_list = [f"{model['name']} - {model['description']}" for model in models_list]
    selected_model_display = st.sidebar.selectbox(
        "Choose a Model",
        options=model_display_list
    )
    if selected_model_display:
        # Extract the actual model name (e.g., "XUNETFS") from the selected string
        selected_model_name = selected_model_display.split(" - ")[0]
else:
    st.sidebar.warning("Could not fetch models from the backend. Please ensure the FastAPI server is running.")

# --- Main Area for Image Upload and Display ---
uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["png", "jpg", "jpeg"])

col1, col2, col3 = st.columns(3)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB") # Ensure image is in RGB for display
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded X-ray", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading uploaded image: {e}")
        uploaded_file = None # Reset uploaded_file to prevent further processing

if uploaded_file and selected_model_name:
    if st.button(f"Process Image with {selected_model_name}", key="process_button"):
        with st.spinner(f"Processing image with {selected_model_name}..."):
            try:
                # Prepare files and data for the POST request
                files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"model_name": selected_model_name}

                # Make the POST request to the FastAPI backend
                response = requests.post(f"{FASTAPI_URL}/process_image", files=files, data=data, timeout=300) # 5 min timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                result = response.json()

                if result.get("success"):
                    prediction_b64 = result.get("prediction_image")
                    bone_only_b64 = result.get("bone_only_image")

                    prediction_img = base64_to_image(prediction_b64)
                    bone_only_img = base64_to_image(bone_only_b64)

                    if prediction_img:
                        with col2:
                            st.subheader("Bone Suppressed")
                            st.image(prediction_img, caption=f"Processed by {selected_model_name}", use_column_width=True)
                    else:
                        with col2:
                            st.error("Could not display bone suppressed image.")

                    if bone_only_img:
                        with col3:
                            st.subheader("Bone Only")
                            st.image(bone_only_img, caption="Subtracted bone map", use_column_width=True)
                    else:
                        with col3:
                            st.error("Could not display bone only image.")
                else:
                    error_message = result.get("error", "An unknown error occurred during processing.")
                    st.error(f"Processing failed: {error_message}")

            except requests.exceptions.Timeout:
                st.error("The processing request timed out. The model might be too slow or the image too large.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error communicating with the processing API: {e}")
            except json.JSONDecodeError:
                st.error("Received an invalid response from the server. Please check the FastAPI server logs.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
elif uploaded_file and not selected_model_name and models_list:
    st.warning("Please select a model from the sidebar to proceed.")
elif not uploaded_file and selected_model_name:
    st.info("Please upload an image to process.")

st.sidebar.markdown("---")
st.sidebar.info(
    "This web application uses a FastAPI backend to perform bone suppression "
    "on chest X-ray images using various deep learning models."
)

# For debugging: show selected model name
# if selected_model_name:
#     st.sidebar.write(f"Selected Model Code: {selected_model_name}")
# if models_list:
#     st.sidebar.json(models_list)

import json # Ensure json is imported if used in get_models error handling
