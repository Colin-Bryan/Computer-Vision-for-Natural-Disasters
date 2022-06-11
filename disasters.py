import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
 
import os
import torch
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchsummary import summary
import urllib.request
import zipfile

#from resnet_model import ResnetModel


## load the pretrained model
#@st.cache()
#def load_model(path: str = 'models/trained_model_resnet50.pt') -> ResnetModel:
   #"""Retrieves the trained model and maps it to the CPU by default,
    #can also specify GPU here."""
    #model = ResnetModel(path_to_pretrained_model=path)
    #return model



## crosswalk for the class labels (from numbers to word)
# crosswalk = {1 : "fire", 2: "flood"}



# Title
st.title("Natural Disasters: Is it a fire or flood?")
st.write("The model classifies: fire, flood, or not disaster from aerial or ground footage")

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