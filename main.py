import streamlit as st
import pandas as pd
import numpy as np
import torch
import io
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F



# Function to display image
def load_image(model):   
    # Create file uploader with streamlit that accepts one image at a time with validation
    uploaded_file = st.file_uploader("", type=["png","jpg","jpeg"], accept_multiple_files = False)

    # If file uploaded
    if uploaded_file is not None:
        # Get image data
        image_data = uploaded_file.getvalue()

        # Open image in UI to display
        st.image(image_data, width = 600)

        # Save image converted to PIL image object
        image = Image.open(io.BytesIO(image_data))

        # Start processing
        st.write('Calculating results...')

        # Run prediction function
        predict(model, image)

# Load model function and return
## Cache model for optimal performance
@st.cache
def load_model():
    # Load saved model from models folder
    model = torch.load("./models/fullmodel.pt", map_location=torch.device('cpu' ))
    # Set model to evaluation mode
    model.eval()
    return model

# Create predict function
def predict(model, image):
    # Create labels_dict and map numbers to type based on training
    labels_dict = {0:'Fire', 1:'Flood', 2:'Not a Disaster'}

    # Prep image to run through model 
    # Create transforms for uploaded images using same transforms as in validation 
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    # Create tensor and preprocess input
    input_tensor = preprocess(image)
    # Create an input batch and add extra dimension to pass through model
    input_batch = input_tensor.unsqueeze(0)

    # Pass image through model
    with torch.no_grad():
        output = model(input_batch)

    # Get probabilities and predictions
    probabilities = F.softmax(output[0], dim=0)
    prediction = np.argmax(probabilities.numpy(), axis=0)

    # Return the prediction and display on UI using label dictionary
    return st.title(labels_dict[prediction])
    

def run():
    ## Create Title
    st.title('Natural Disaster Identification')
    # Create Subheader
    st.subheader("Upload an Image")

    # Load Model
    model = load_model()

    # Load image
    image = load_image(model)

# Start
if __name__ == '__main__':
    run()
