import streamlit as st
import pandas as pd
import numpy as np
import torch
import io
import cv2
from PIL import Image
from torchvision import transforms


# Function to display image
def load_image():   
    # Create file uploader with streamlit that accepts one image at a time with validation
    uploaded_file = st.file_uploader("", type=["png","jpg","jpeg"], accept_multiple_files = False)

    # If file uploaded
    if uploaded_file is not None:
        # Get image data
        image_data = uploaded_file.getvalue()
        # Open image in UI to display
        st.image(image_data)
        # Return image_data converted to PIL image object
        return Image.open(io.BytesIO(image_data))

# Load model function and return
def load_model():
    # Load saved model from models folder
    model = torch.load("./models/fullmodel.pt", map_location=torch.device('cpu' ))
    # Set model to evaluation mode
    model.eval()
    return model

# Create predict function
def predict(model, image):
    # Create labels and map to numbers
    labels = {0:'fire', 1:'flood', 2:'not_disaster'}

    # Prep image to run through model

######### Need to do the same thing we are doing to images when running through our model ############
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
#######################################################################################################
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

######### Need to run the same predictions function we are using ######################################
    with torch.no_grad():
        output = model(input_batch)
#######################################################################################################
    return output

def run():
    ## Create Title
    st.title('Natural Disaster Identification')
    # Create Subheader
    st.subheader("Upload Single or Multiple Images")

    # Load Model
    model = load_model()

    # Load image
    image = load_image()

    # Create button called result that starts prediction
    result = st.button('Run on image')

    # If result is clicked
    if result:
        st.write('Calculating results...')

        # Run prediction
        predict(model, image)

# Start
if __name__ == '__main__':
    run()