import os
import pandas as pd
import cv2
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from pandas.core.frame import DataFrame

AIDER_TRAIN_PATH =  os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/raw/AIDER_filtered/train'))
AIDER_VAL_PATH =  os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/raw/AIDER_filtered/val'))
MEDIC_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/raw/data_disaster_types'))
PROCESSED_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/processed'))

class CustomDataset(Dataset):
    '''
    Custom PyTorch Dataset for image classification
    Must contain 3 parts: __init__, __len__ and __getitem__
    '''

    def __init__(self, labels_df: DataFrame, data_dir: str, class_mapper: dict, transform=None):
        '''
        Args:
            labels_df (DataFrame): Dataframe containing the image names and corresponding labels
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
        # Load the image
        img_path = os.path.join(self.data_dir,
                                self.labels_df.iloc[idx, 0])

        # For a normal image file (jpg,png) use the below
        image = cv2.imread(img_path) # Use this for normal color images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Use this for color images to rearrange channels BGR -> RGB
        image = Image.fromarray(image) # convert numpy array to PIL image

        # Load the label
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
        classes = ['fire','flood', 'not_disaster']
        idx_to_class = {i:j for i,j in enumerate(classes)}
        class_to_idx = {v:k for k,v in idx_to_class.items()}

        data_transforms = self.get_data_transforms()

        train_dataset = CustomDataset(labels_df=train_df,
                                data_dir=os.getcwd(),
                                class_mapper=class_to_idx,
                                transform = data_transforms['train'])

        val_dataset = CustomDataset(labels_df=val_df,
                                data_dir=os.getcwd(),
                                class_mapper=class_to_idx,
                                transform = data_transforms['val'])
        return train_dataset, val_dataset

    def save_dataloader(self, dataloader, file_prefix):
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
        dataset_sizes = {'train':len(train_dataset),'val':len(val_dataset)}
        return dataset_sizes

    def filter_for_existing_paths(self, df):
        print(f'Number of rows before filtering for existing paths: {len(df)}')
        all_paths = df[['image_path']].values.tolist()
        non_existent_paths = []
        for path in all_paths:
            path = path[0]
            full_path = os.path.join(MEDIC_PATH, path)
            if not os.path.exists(full_path):
                non_existent_paths.append(path)
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
        filtered_df = df[condition_1 | condition_2 |  condition_3]
        filtered_df = filtered_df[['image_path', 'class_label']]
        final_df = self.filter_for_existing_paths(filtered_df)
        return final_df

    def combine_df(self, df, type_dataset):
        AIDER_range = {
            'train': {
                'flood': [101, 526], #0
                'fire': [101,521], #0
                'normal': [1001,4390]
            },
            'val': {
                'flood': [1, 100], #0001 to 0100
                'fire': [1,100], #0001 to 0100
                'normal': [1,1000] # 0001
            }
        }
        ranges = AIDER_range[type_dataset]
        all_info = []
        for disaster_type in ranges: # range is flood, fire, normal
            if disaster_type == 'normal':
                normalized_disaster_type = 'not_disaster'
            else:
                normalized_disaster_type = disaster_type
            lower = ranges[disaster_type][0]
            upper = ranges[disaster_type][1]
            for index in range(lower, upper+1): # for each disaster type
                str_index = str(index)
                num_zeros_before = 4 - len(str_index)
                str_num = '0' * num_zeros_before + str(index)
                base_path = AIDER_TRAIN_PATH if type_dataset == 'train' else AIDER_VAL_PATH
                image_path = os.path.join(base_path, f'{disaster_type}/{disaster_type}_image{str_num}.jpg')
                all_info.append([image_path, normalized_disaster_type]) 
        new_df = pd.DataFrame(all_info, columns=['image_path', 'class_label'])
        combined_df = pd.concat([df, new_df])
        return combined_df

    def write_df_as_tsv(self, df, df_name, filename):
        path = os.path.join(PROCESSED_DATA_PATH, filename)
        df.to_csv(path, sep="\t")
        print(f'Wrote {df_name} to {PROCESSED_DATA_PATH}')

    def generate_aider_datasets(self):
        data_transforms = self.get_data_transforms()
        aider_train_dataset = datasets.ImageFolder(AIDER_TRAIN_PATH, data_transforms['train'])
        aider_val_dataset= datasets.ImageFolder(AIDER_VAL_PATH, data_transforms['val'])
        return aider_train_dataset, aider_val_dataset

    def generate_medic_datasets(self):
        medic_train_df = self.generate_medic_filtered_df('train')
        self.write_df_as_tsv(medic_train_df, 'medic_train_df', 'filtered_medic_train.tsv')
        medic_val_df = self.generate_medic_filtered_df('test')
        self.write_df_as_tsv(medic_val_df, 'medic_val_df', 'filtered_medic_val.tsv')
        medic_train_dataset, medic_val_dataset = self.init_custom_dataset(medic_train_df, medic_val_df)
        return medic_train_dataset, medic_val_dataset, medic_train_df, medic_val_df

    def generate_combined_datasets(self, medic_train_df, medic_val_df):
        combined_train_df = self.combine_df(medic_train_df, 'train')
        self.write_df_as_tsv(combined_train_df, 'combined_train_df', 'combined_train.tsv')
        combined_val_df = self.combine_df(medic_val_df, 'val')
        self.write_df_as_tsv(combined_val_df, 'combined_val_df', 'combined_val.tsv')
        combined_train_dataset, combine_val_dataset = self.init_custom_dataset(combined_train_df, combined_val_df)
        return combined_train_dataset, combine_val_dataset

def main():
    custom_dataloader = CustomDataloader()
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

    for data_source in data_sources:
        train_dataset = data_sources[data_source]['train']
        val_dataset =  data_sources[data_source]['val']
        dataset_sizes = custom_dataloader.init_dataloaders(train_dataset, val_dataset, batch_size, data_source)
        print(f'{data_source}: {dataset_sizes}')

if __name__ == "__main__":
    main()
