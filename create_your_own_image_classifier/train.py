import argparse
import os
import sys
import json

import torch
from torch import nn # Neural Network tools
from torch import optim # For optimizer
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from workspace_utils import keep_awake, active_session

from collections import OrderedDict

from PIL import Image
import numpy as np

arg_parser = argparse.ArgumentParser(description='Pass in the necessary arguments to train your network')

arg_parser.add_argument('data_directory', help='The path to the data directory containing images in a format PyTorch expects')
arg_parser.add_argument('--save_dir', action='store', help="Directory to save checkpoints, default='checkpoint_dir'", default='checkpoint_dir')
arg_parser.add_argument('--arch', action='store', help="Choice of network architecture - vgg16 or vgg13, default='vgg16'", default='vgg16', choices={"vgg13", "vgg16"})
arg_parser.add_argument('--learning_rate', action='store', type=float, help="Learning rate, default=0.003", default=0.003)
arg_parser.add_argument('--hidden_units', action='store', type=int, help="Number of hidden units in hidden layer between 25088 and 102, default=1664", default=1664)
arg_parser.add_argument('--epochs', action='store', type=int, help="Numer of epochs to be used to train the network, default=20", default=20)
arg_parser.add_argument('--gpu', action='store_true', default=False, help="To use GPU for training or not")

args = arg_parser.parse_args()
print(args)

data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

data_transforms = {'train': [transforms.RandomRotation(30), 
                             transforms.RandomResizedCrop(224), 
                             transforms.RandomHorizontalFlip(), 
                             transforms.ToTensor(), 
                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                  [0.229, 0.224, 0.225])],
                   'valid_test': [transforms.Resize(255), 
                   transforms.CenterCrop(224), 
                   transforms.ToTensor(), 
                   transforms.Normalize([0.485, 0.456, 0.406], 
                                        [0.229, 0.224, 0.225])]}

image_datasets = {'train': datasets.ImageFolder(train_dir, transform=transforms.Compose(data_transforms['train'])), 
                  'valid': datasets.ImageFolder(valid_dir, transform=transforms.Compose(data_transforms['valid_test'])), 
                  'test': datasets.ImageFolder(test_dir, transform=transforms.Compose(data_transforms['valid_test']))}

dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True), 
               'valid': DataLoader(image_datasets['valid'], batch_size=64), 
               'test': DataLoader(image_datasets['test'], batch_size=64)}

model = models.vgg16(pretrained=True) if args.arch == 'vgg16' else models.vgg13(pretrained=True)

# Changing the architecture from that in the notebook as the user is allowed to enter hidden units just once i.e. only for one layer
# Both vgg13 and vgg16 have same number of inputs i.e. to the classifier
classifier_layers = OrderedDict([('hidden_1', nn.Linear(25088, 7168)), 
                                 ('relu_1', nn.ReLU()),
                                 ('dropout_1', nn.Dropout(p=0.2)),
                                 ('hidden_2', nn.Linear(7168, args.hidden_units)), 
                                 ('relu_2', nn.ReLU()),
                                 ('dropout_2', nn.Dropout(p=0.2)),
                                 ('output', nn.Linear(args.hidden_units, 102)),
                                 ('softmax', nn.LogSoftmax(dim=1))])

classifier = nn.Sequential(classifier_layers)
model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Training method

def train_validate_network(epochs=args.epochs, device=device):
    model.to(device)
    best_model = {'validation_accuracy': 0.0, 'model_dict': model.state_dict(), 'epoch': -1}
    running_loss = 0
    for epoch in range(epochs):
        n_images = 0
        for images, labels in dataloaders['train']:
            n_images += images.shape[0]
            print("Epoch: {}, Images: {}".format(epoch+1, n_images))
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad() # To prevent stacking of gradients over iterations
            
            # Forward pass and BackProp
#             images = images.view(images.shape[0], -1)
#             print(images.shape)
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print("Trained on {} images.".format(n_images))
            # Validation
            valid_running_loss = 0
            accuracy = 0
            model.eval() # Disable Dropout
            with torch.no_grad():
                for images, labels in dataloaders['valid']:
                    images, labels = images.to(device), labels.to(device)
                    logps = model.forward(images)
                    batch_loss = criterion(logps, labels)

                    valid_running_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    _, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            valid_dataloader_len = len(dataloaders['valid'])
            train_loss = running_loss/n_images
            valid_loss = valid_running_loss/valid_dataloader_len
            valid_accuracy = accuracy/valid_dataloader_len
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {train_loss:.3f}.. "
                  f"Validation loss: {valid_loss:.3f}.. "
                  f"Validation accuracy: {valid_accuracy:.3f}")
                
            if valid_accuracy > best_model['validation_accuracy']:
                best_model['validation_accuracy'] = valid_accuracy
                best_model['model_dict'] = model.state_dict()
                best_model['epoch'] = epoch + 1
                print("The best model so far had a validation accuracy of {}, trained in {}th epoch.".format(valid_accuracy, epoch + 1))
                running_loss = 0
                model.train()
    return best_model

best_model = None

with active_session():
    best_model = train_validate_network(args.epochs, device)
 
# If the best model isn't the one trained in the last epoch
model.load_state_dict(best_model['model_dict'])

# Commented as testing the network against testing data isn't part of the requirement/rubric

# model.eval() 
# test_running_loss = 0
# accuracy = 0
# with torch.no_grad():
#     for images, labels in dataloaders['test']:
#         images, labels = images.to(device), labels.to(device)
#         logps = model.forward(images)
#         batch_loss = criterion(logps, labels)

#         test_running_loss += batch_loss.item()

#         # Calculate accuracy
#         ps = torch.exp(logps)
#         _, top_class = ps.topk(1, dim=1)
#         equals = top_class == labels.view(*top_class.shape)
#         accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

# test_dataloader_len = len(dataloaders['test'])
# test_loss = test_running_loss/test_dataloader_len
# test_accuracy = accuracy/test_dataloader_len
# print(f"Test loss: {test_loss:.3f}.. "
#       f"Test accuracy: {test_accuracy:.3f}")

model = model.to('cpu')
checkpoint = {'input_size': 25088,
              'output_size': 102,
              'hidden_layers': [each.out_features for each in classifier if isinstance(each, torch.nn.modules.linear.Linear)][:-1],
              'state_dict': model.state_dict(),
              'validation_accuracy': best_model['validation_accuracy'],
              'class_to_index_mapping': image_datasets['train'].class_to_idx, 
              'architecture': args.arch}

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
torch.save(checkpoint, args.save_dir + '/checkpoint.pth')

print('Use predict.py to do predictions on this trained model now!')