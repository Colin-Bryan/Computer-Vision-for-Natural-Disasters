# Identifying Disasters from Different Perspectives

#### Category: Social Media & News
#### Type: Computer Vision

#### Links to original datasets

- [AIDER (Aerial Image Dataset for Emergency Response Applications)](https://zenodo.org/record/3888300#.YqkdjOjMKUl)
- [MEDIC (Multi-Task Learning for Disaster Image Classification)](https://crisisnlp.qcri.org/crisis-image-datasets-asonam20)

## Getting started
1. Download the following files to data/raw folder
- [MEDIC (data_disaster_types)](https://crisisnlp.qcri.org/crisis-image-datasets-asonam20)
Download the 3.2G dataset labelled "Disaster types" towards the bottom of the page.
- [AIDER_filtered](https://drive.google.com/file/d/15w4mdKR9LHjPc5WCeUcswoI34_pzj41r/view?usp=sharing)
2. Install the requirements needed to run the python scripts.
```
pip install -r requirements.txt
```
3. Extract from AIDER_filtered.zip and data_disaster_types.tar.gz
```
python scripts/make_datasets.py
```
4. Create and save the dataloaders and tsv files from intermediate steps
```
python scripts/build_features.py
```
5. Train model and predict
```
python scripts/model.py
```
Please note that model.py may need to be run on GPU.