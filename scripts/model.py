import os
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
from pandas.core.frame import DataFrame
import time
import copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torchvision
import torch.nn.functional as F

# base folder paths that will be joined with relevant filenames to create full paths
PROCESSED_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/processed'))
MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'models'))

class Model:    
    #Model Architecture: ResNet50 with transfer learning
    def instantiate_train(self, net, dataloaders, dataset_sizes, device):
        # Shut off autograd for all layers to freeze model so the layer weights are not trained
        for param in net.parameters():
            param.requires_grad = False

        # Get the number of inputs to final Linear layer
        num_ftrs = net.fc.in_features

        # Replace final Linear layer with a new Linear with the same number of inputs but just 3 outputs,
        # since we have 3 classes - fire, flood and normal or diaster
        net.fc = nn.Linear(num_ftrs, 3)

        # Cross entropy loss combines softmax and nn.NLLLoss() in one single class.
        criterion = nn.CrossEntropyLoss()

        # Define optimizer
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # Learning rate scheduler - decay LR by a factor of 0.1 every 7 epochs
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Train the model
        net = self.train_model(net, criterion, optimizer, dataloaders, lr_scheduler, device, dataset_sizes)

    def train_model(self, model, criterion, optimizer, dataloaders, scheduler, device, dataset_sizes, num_epochs=25):
        model = model.to(device) # Send model to GPU if available
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Get the input images and labels, and send to GPU if available
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the weight gradients
                    optimizer.zero_grad()

                    # Forward pass to get outputs and calculate loss
                    # Track gradient only for training data
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backpropagation to get the gradients with respect to each weight
                        # Only if in train
                        if phase == 'train':
                            loss.backward()
                            # Update the weights
                            optimizer.step()

                    # Convert loss into a scalar and add it to running_loss
                    running_loss += loss.item() * inputs.size(0)
                    # Track number of correct predictions
                    running_corrects += torch.sum(preds == labels.data)

                # Step along learning rate scheduler when in train
                if phase == 'train':
                    scheduler.step()

                # Calculate and display average loss and accuracy for the epoch
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # If model performs better on val set, save weights as the best model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:3f}'.format(best_acc))

        # Load the weights from best model
        model.load_state_dict(best_model_wts)

        return model

    def test_model(self, model, test_loader, device):
        model = model.to(device)
        # Turn autograd off
        with torch.no_grad():

            # Set the model to evaluation mode
            model.eval()

            # Set up lists to store true and predicted values
            y_true = []
            test_preds = []

            # Calculate the predictions on the test set and add to list
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                # Feed inputs through model to get raw scores
                logits = model.forward(inputs)
                # Convert raw scores to probabilities (not necessary since we just care about discrete probs in this case)
                probs = F.softmax(logits,dim=1)
                # Get discrete predictions using argmax
                preds = np.argmax(probs.numpy(),axis=1)
                # Add predictions and actuals to lists
                test_preds.extend(preds)
                y_true.extend(labels)

            # Calculate the accuracy
            test_preds = np.array(test_preds)
            y_true = np.array(y_true)
            test_acc = np.sum(test_preds == y_true)/y_true.shape[0]
            
            # Recall for each class
            recall_vals = []
            for i in range(3):
                class_idx = np.argwhere(y_true==i)
                total = len(class_idx)
                correct = np.sum(test_preds[class_idx]==i)
                recall = correct / total
                recall_vals.append(recall)
        
        return test_acc, recall_vals

    def savemodel(self, net):
        #Save the entire model
        filename = 'fullmodel.pt'
        path = os.path.join(MODELS_PATH, filename)
        torch.save(net, path)

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
    # load dataloaders and get dataset sizes from tsv files
    combined_train_dataloader_path = os.path.join(PROCESSED_DATA_PATH, 'combined_train_dataloader.pkl')
    train_dataloader = torch.load(combined_train_dataloader_path)
    combined_valdata_loader_path = os.path.join(PROCESSED_DATA_PATH, 'combined_val_dataloader.pkl')
    val_dataloader = torch.load(combined_valdata_loader_path)

    # Set up dict for dataloaders
    dataloaders = {'train':train_dataloader,'val':val_dataloader}
    combined_train_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH ,'combined_train.tsv'), sep='\t', header=0)
    combined_val_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH ,'combined_val.tsv'), sep='\t', header=0)

    # Store size of training and validation sets
    dataset_sizes = {'train': len(combined_train_df),'val': len(combined_val_df)}
    models = [torchvision.models.resnet50(pretrained=True)]
    # labels for combined dataset
    classes = ['fire','flood', 'not_disaster']
    
    model_obj = Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in models:
        model_obj.instantiate_train(model, dataloaders, dataset_sizes, device)
        acc, recall_vals = model_obj.test_model(model, val_dataloader, device)
        print('Test set accuracy is {:.3f}'.format(acc))
        for i in range(3):
            print('For class {}, recall is {}'.format(classes[i], recall_vals[i]))
        model_obj.savemodel(model)

if __name__ == "__main__":
    main()
