from unet_model import UNet
import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_classes=1).to(device)
model.load_state_dict(torch.load("src/best_model.pth", map_location=device))
model.eval()

# --- Streamlit UI ---
st.title("Brain MRI Segmentation (U-Net)")
st.write("Upload an MRI image to segment the tumor region.")

uploaded_file = st.file_uploader(
    "Choose an MRI image", type=["jpg", "png", "jpeg", "tif"]
)

if uploaded_file is not None:
    # Read and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = np.array(image)
    img_resized = cv2.resize(img, (128, 128))
    img_norm = img_resized / 255.0
    img_transposed = np.transpose(img_norm, (2, 0, 1))
    input_tensor = (
        torch.tensor(img_transposed, dtype=torch.float32).unsqueeze(0).to(device)
    )

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output).cpu().squeeze().numpy()
        pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255

    # Show results
    st.subheader("Predicted Mask")
    st.image(pred_mask_bin, caption="Segmentation Mask", use_container_width=True)
