import streamlit as st
import pandas as pd
import numpy as np
import torch
import io
import cv2
from PIL import Image
from torchvision import transforms

# Function to display image
def load_image(model):   
    # Create file uploader with streamlit that accepts multiple images at once with validation
    uploaded_files= st.file_uploader("", type=["png","jpg","jpeg"], accept_multiple_files = True)

    # If file(s) uploaded
    if uploaded_files is not None:
        # Loop through each
        for image_file in uploaded_files:
            # Get image data
            image_data = image_file.getvalue()

            # Open image in UI
            st.image(image_data)

            # Run image through model and make predictions
            predict(model, image_data)

# Each user interation with page causes streamlit to run the python file
# top down. Therefore, use the cache() decorator to prevent streamlit
# from reloading the model each time a user uploads and image

st.cache()
def load_model():
    model = torch.load("./models/fullmodel.pt", map_location=torch.device('cpu' ))
    return model

def predict(model, image):
    image_prep = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
    # need to add code for data transformers

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Use this for color images to rearrange channels BGR -> RGB
    image = Image.fromarray(image) 
    transformed_image = image_prep(image)

    with torch.no_grad():

        model.eval()
    
        prediction = model(transformed_image)
        st.write(f"The predicted value for this image is... {prediction}")
    

def run():
    ## Create Title
    st.title('Disaster Identification')
    # Create Subheader
    st.subheader("Upload Single or Multiple Images")

    # Load Model
    model = load_model()

    # Load labels
    #categories = load_labels()

    # Load image
    load_image(model)

    # Load result
    #result = st.button('Run on image')
    #if result:
    #    st.write('Calculating results...')
    #    predict(model, image)


if __name__ == '__main__':
    run()