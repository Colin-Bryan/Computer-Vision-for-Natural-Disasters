# Identifying Disasters from Different Perspectives

#### Category: Social Media & News
#### Type: Computer Vision

### Motivation
- Classify images by disaster type to understand how neural network applications can lead to valuable tools for emergency response and disaster management applications.
- Deploy a neural network model as a proof of concept to identify a subset of natural disasters regardless of image vantage point. To limit the scope of our model for this proof of concept, we focused on classifying images as a fire, flood, or non-disaster/normal.

We used images from AIDER (Aerial Image Dataset for Emergency Response Applications) and MEDIC (Multi-Task Learning for Disaster Image Classification), and we used Streamlit to create a front-end that can accept one image and classify according to our trained model.

### Links to original datasets
- [AIDER](https://zenodo.org/record/3888300#.YqkdjOjMKUl)
- [MEDIC](https://crisisnlp.qcri.org/crisis-image-datasets-asonam20)
### Links to papers
- [AIDER](https://openaccess.thecvf.com/content_CVPRW_2019/papers/UAVision/Kyrkou_Deep-Learning-Based_Aerial_Image_Classification_for_Emergency_Response_Applications_Using_Unmanned_CVPRW_2019_paper)
- [MEDIC](https://arxiv.org/pdf/2108.12828.pdf)

## Use Streamlit to classify an image
1. Install the requirements needed to use Streamlit
```
pip install -r requirements.txt
```
2. Start the Streamlit app
```
streamlit run main.py
```

## Build features and train model
1. Download the following files to the data/raw folder
- [MEDIC (data_disaster_types)](https://crisisnlp.qcri.org/crisis-image-datasets-asonam20):
Download the 3.2G dataset labelled "Disaster types" towards the bottom of the page.
- [AIDER_filtered](https://drive.google.com/file/d/15w4mdKR9LHjPc5WCeUcswoI34_pzj41r/view?usp=sharing):
Please note that the structure and content of the original AIDER dataset has been modified to create AIDER_filtered.
2. Install the requirements needed to run the python scripts. (If you already ran the command before using Streamlit as specified above, please skip this step.)
```
pip install -r requirements.txt
```
3. Extract from AIDER_filtered.zip and data_disaster_types.tar.gz (MEDIC dataset)
```
python scripts/make_datasets.py
```
4. Create and save the dataloaders and tsv files from intermediate steps
```
python scripts/build_features.py
```
5. Train model, test model, and get metrics
```
python scripts/model.py
```
We are using the ResNet-50 pre-trained model. Please note that model.py may need to be run on GPU.
