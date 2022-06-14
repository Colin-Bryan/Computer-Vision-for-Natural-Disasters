import os
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
from pandas.core.frame import DataFrame

PROCESSED_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/processed'))

class CustomDataset(Dataset):
    # Please note that this class is the same as the one from build_features.py
    '''
    Custom PyTorch Dataset for image classification
    Must contain 3 parts: __init__, __len__ and __getitem__
    Used for MEDIC and combined dataset
    '''

    def __init__(self, labels_df: DataFrame, data_dir: str, class_mapper: dict, transform=None):
        '''
        Args:
            labels_df (DataFrame): Dataframe containing the image names (index 0) and corresponding labels (index 1)
            data_dir (string): Path to directory containing the images
            class_mapper (dict): Dictionary mapping string labels to numeric labels
            transform (callable,optional): Optional transform to be applied to images
        '''
        self.labels_df = labels_df
        self.transform = transform
        self.data_dir = data_dir
        self.classes = self.labels_df['class_label'].unique()
        self.classmapper = class_mapper

    def __len__(self):
        '''Returns the number of images in the Dataset'''
        return len(self.labels_df)

    def __getitem__(self, idx):
        '''
        Returns the image and corresponding label for an input index
        Used by PyTorch to create the iterable DataLoader

        Args:
            idx (integer): index value for which to get image and label
        '''
        # Load the image: join data/raw folder path with image_path from labels_df (self.labels_df.iloc[idx, 0])
        #   image_path should start with 'data_disaster_types' or 'AIDER_filtered'
        img_path = os.path.join(self.data_dir,
                                self.labels_df.iloc[idx, 0])

        # For a normal image file (jpg,png) use the below
        image = cv2.imread(img_path) # Use this for normal color images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Use this for color images to rearrange channels BGR -> RGB
        image = Image.fromarray(image) # convert numpy array to PIL image

        # Load the label: 'fire', 'flood', or 'not_disaster'
        label = self.labels_df.iloc[idx, 1]
        label = self.classmapper[label]

        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

def main():
    # load dataloaders and get dataset sizes from length of tsv files
    combined_dataloader_path = os.path.join(PROCESSED_DATA_PATH, 'combined_train_dataloader.pkl')
    train_dataloader = torch.load(combined_dataloader_path)

if __name__ == "__main__":
    main()