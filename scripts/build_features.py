import os
import pandas as pd
import cv2
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from pandas.core.frame import DataFrame

# base folder paths that will be joined with relevant filenames to create full paths
AIDER_TRAIN_PATH =  os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/raw/AIDER_filtered/train'))
AIDER_VAL_PATH =  os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/raw/AIDER_filtered/val'))
MEDIC_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/raw/data_disaster_types'))
RAW_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/raw'))
PROCESSED_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/processed'))

class CustomDataset(Dataset):
    '''
    Custom PyTorch Dataset for image classification
    Must contain 3 parts: __init__, __len__ and __getitem__
    Used for MEDIC (data_disaster_types) and combined dataset
    '''

    def __init__(self, labels_df: DataFrame, data_dir: str, class_mapper: dict, transform=None):
        '''
        Args:
            labels_df (DataFrame): Dataframe containing
                'image_path' column (index 0): values start with 'AIDER_filtered' or 'data_disaster_types'
                'class_label' column (index 1) for 'fire', 'flood', or 'not_disaster'
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

class CustomDataloader:
    # Set up transformations for training and validation (test) data
    # For training data we will do randomized cropping to get to 224 * 224, randomized horizontal flipping, and normalization
    # For test set we will do only center cropping to get to 224 * 224 and normalization
    def get_data_transforms(self):
        data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        }
        return data_transforms

    def init_custom_dataset(self, train_df, val_df):
        # classes for MEDIC/combined dataset
        classes = ['fire','flood', 'not_disaster']
        idx_to_class = {i:j for i,j in enumerate(classes)}
        class_to_idx = {v:k for k,v in idx_to_class.items()}

        data_transforms = self.get_data_transforms()

        train_dataset = CustomDataset(labels_df=train_df,
                                data_dir=RAW_DATA_PATH,
                                class_mapper=class_to_idx,
                                transform = data_transforms['train'])

        val_dataset = CustomDataset(labels_df=val_df,
                                data_dir=RAW_DATA_PATH,
                                class_mapper=class_to_idx,
                                transform = data_transforms['val'])
        return train_dataset, val_dataset

    def save_dataloader(self, dataloader, file_prefix):
        # save dataloaders to use in model.py
        filename = f'{file_prefix}_dataloader.pkl'
        dataloader_path = os.path.join(PROCESSED_DATA_PATH, filename)
        torch.save(dataloader, dataloader_path)
        print(f'Saved {filename} to {PROCESSED_DATA_PATH}')

    # Create DataLoaders for training and validation sets
    #num_workers:tells dataloader instane how many sub-processes to use for data loading. If zero, GPU has to weight for CPU
    #load data.greater num_workers more efficiently the CPU load data and less the GPU has to wait. 
    #Google Colab: suggested num_workers=2
    def init_dataloaders(self, train_dataset, val_dataset, batch_size, data_source):
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=2)
        self.save_dataloader(train_loader, f'{data_source}_train')

        val_loader = DataLoader(val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=2)
        self.save_dataloader(val_loader, f'{data_source}_val')
        # Store size of training and validation sets
        dataset_sizes = {'train': len(train_dataset),'val': len(val_dataset)}
        return dataset_sizes

    def filter_for_existing_paths(self, df):
        print(f'Number of rows before filtering for existing paths: {len(df)}')
        all_paths = df['image_path'].values.tolist()
        non_existent_paths = []
        for path in all_paths:
            full_path = os.path.join(MEDIC_PATH, path)
            if not os.path.exists(full_path):
                non_existent_paths.append(path)
        # keep rows that contain image_paths that correspond to existing images
        df = df[~df['image_path'].isin(non_existent_paths)]
        print(f'Number of rows after filtering for existing paths: {len(df)}')
        return df

    def generate_medic_filtered_df(self, type_dataset):
        tsv_file_path = os.path.join(MEDIC_PATH, f'consolidated_disaster_types_{type_dataset}_final.tsv')
        df = pd.read_csv(tsv_file_path, sep='\t', header=0)
        print(f'Number of rows from original MEDIC {type_dataset} file: {len(df)}')
        condition_1 = df['class_label'] == 'flood'
        condition_2 = df['class_label'] == 'fire'
        condition_3 = df['class_label'] == 'not_disaster'
        # filter for rows that have labels 'flood', 'fire', or 'not_disaster'
        filtered_df = df[condition_1 | condition_2 |  condition_3]
        # need 'image_path' and 'class_label' columns only for custom dataset class
        filtered_df = filtered_df[['image_path', 'class_label']]
        # ensure that rows contain image_path values that correspond to existing images
        filtered_df = self.filter_for_existing_paths(filtered_df)
        # add MEDIC folder name after filtering to differentiate data sources before combining
        filtered_df['image_path'] = 'data_disaster_types/' + filtered_df['image_path'].astype(str)
        return filtered_df

    def combine_df(self, df, type_dataset):
        # Please note that the functionality works only if the images from the 
        #   AIDER folder are split into train and val folders this way
        AIDER_range = {
            'train': { 
                'flood': [101, 526], # consecutive values from 101 to 526:'flood_image0101.jpg' to 'flood_image0526.jpg'
                'fire': [101, 521], # consecutive values from 101 to 526:'fire_image0101.jpg' to 'fire_image0521.jpg'
                'normal': [1001, 4390] # consecutive values from 1001 to 4390: 'normal_image1001.jpg' to 'normal_image4390.jpg'
            },
            'val': {
                'flood': [1, 100], # consecutive values from 0001 to 0100: 'flood_image0001.jpg' to 'flood_image0100.jpg'
                'fire': [1, 100], # consecutive values from 0001 to 0100: 'fire_image0001.jpg' to 'fire_image0100.jpg'
                'normal': [1, 1000] # consecutive values from 0001 to 1000: 'normal_image0001.jpg' to 'normal_image1000.jpg'
            }
        }
        ranges = AIDER_range[type_dataset]
        all_info = []
        for disaster_type in ranges:
            if disaster_type == 'normal':
                # standardize label names across data sources
                normalized_disaster_type = 'not_disaster'
            else:
                # 'flood' and 'fire' labels are constant between AIDER_filtered and MEDIC
                normalized_disaster_type = disaster_type
            lower = ranges[disaster_type][0] # lower numerical bound in folder of type_dataset for disaster_type
            upper = ranges[disaster_type][1] # upper numerical bound in folder of type_dataset for disaster_type
            for index in range(lower, upper+1): # for each image with number index
                str_index = str(index)
                # The numerical portion of each image name all have four digits
                num_zeros_before = 4 - len(str_index)
                str_num = '0' * num_zeros_before + str_index
                # add data source folder name to differentiate between MEDIC images and to append data/raw to get full path
                image_path = f'AIDER_filtered/{type_dataset}/{disaster_type}/{disaster_type}_image{str_num}.jpg'
                all_info.append([image_path, normalized_disaster_type]) 
        # create new dataframe for AIDER
        new_df = pd.DataFrame(all_info, columns=['image_path', 'class_label'])
        # combine MEDIC and AIDER dataframes
        combined_df = pd.concat([df, new_df])
        return combined_df

    def write_df_as_tsv(self, df, df_name, filename):
        # save dataframe as tsv files to use in model.py
        path = os.path.join(PROCESSED_DATA_PATH, filename)
        df.to_csv(path, sep="\t")
        print(f'Wrote {df_name} to {PROCESSED_DATA_PATH}')

    def generate_aider_datasets(self):
        data_transforms = self.get_data_transforms()
        aider_train_dataset = datasets.ImageFolder(AIDER_TRAIN_PATH, data_transforms['train'])
        aider_val_dataset= datasets.ImageFolder(AIDER_VAL_PATH, data_transforms['val'])
        return aider_train_dataset, aider_val_dataset

    def generate_medic_datasets(self):
        # generate medic_train_df and medic_val_df in preparation for using custom dataset class
        medic_train_df = self.generate_medic_filtered_df('train')
        # save intermediate tsv files to use in model.py
        self.write_df_as_tsv(medic_train_df, 'medic_train_df', 'filtered_medic_train.tsv')
        medic_val_df = self.generate_medic_filtered_df('test')
        self.write_df_as_tsv(medic_val_df, 'medic_val_df', 'filtered_medic_val.tsv')
        medic_train_dataset, medic_val_dataset = self.init_custom_dataset(medic_train_df, medic_val_df)
        return medic_train_dataset, medic_val_dataset, medic_train_df, medic_val_df

    def generate_combined_datasets(self, medic_train_df, medic_val_df):
        # generate combined_train_df and combined_val_df in preparation for using custom dataset class
        combined_train_df = self.combine_df(medic_train_df, 'train')
        # save intermediate tsv files to use in model.py
        self.write_df_as_tsv(combined_train_df, 'combined_train_df', 'combined_train.tsv')
        combined_val_df = self.combine_df(medic_val_df, 'val')
        self.write_df_as_tsv(combined_val_df, 'combined_val_df', 'combined_val.tsv')
        combined_train_dataset, combine_val_dataset = self.init_custom_dataset(combined_train_df, combined_val_df)
        return combined_train_dataset, combine_val_dataset

def main():
    custom_dataloader = CustomDataloader()
    # generate datasets for the data sources
    aider_train_dataset, aider_val_dataset = custom_dataloader.generate_aider_datasets()
    medic_train_dataset, medic_val_dataset, medic_train_df, medic_val_df = custom_dataloader.generate_medic_datasets()
    combined_train_dataset, combined_val_dataset = custom_dataloader.generate_combined_datasets(medic_train_df, medic_val_df)

    data_sources = {
      'AIDER': {
          'train': aider_train_dataset,
          'val': aider_val_dataset
      },
      'MEDIC': {
          'train': medic_train_dataset,
          'val': medic_val_dataset
      },
      'combined': {
          'train': combined_train_dataset,
          'val': combined_val_dataset
      }
    }
    batch_size = 4

    for data_source in data_sources: # 'AIDER', 'MEDIC', or 'combined'
        train_dataset = data_sources[data_source]['train']
        val_dataset =  data_sources[data_source]['val']
        dataset_sizes = custom_dataloader.init_dataloaders(train_dataset, val_dataset, batch_size, data_source)
        print(f'{data_source}: {dataset_sizes}')

if __name__ == "__main__":
    main()
