import streamlit as st
import pandas as pd
import numpy as np
import io
from PIL import Image

# Function to display image
def load_image():   
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
            predict(image_data)

def load_model():
    return model

def predict(image):
    Image.open(io.BytesIO(image))
    

def run():
    ## Create Title
    st.title('Natural Disaster Identification')
    # Create Subheader
    st.subheader("Upload Single or Multiple Images")

    # Load Model
    #model = load_model()

    # Load labels
    #categories = load_labels()

    # Load image
    load_image()

    # Load result
    #result = st.button('Run on image')
    #if result:
    #    st.write('Calculating results...')
    #    predict(model, categories, image)


if __name__ == '__main__':
    run()