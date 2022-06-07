import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Function to display image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Page Layout
## Create Title
st.title('Natural Disaster Identification')

st.subheader("Upload Single or Multiple Images")
uploaded_files= st.file_uploader("",type=["png","jpg","jpeg"], 
                                    accept_multiple_files = True)

if uploaded_files is not None:
        # To See details
        for image_file in uploaded_files:
            file_details = {"filename":image_file.name,"filetype":image_file.type,
                            "filesize":image_file.size}
            st.write(file_details)
            st.image(load_image(image_file), width=250)
