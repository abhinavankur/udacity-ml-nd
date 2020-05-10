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

arg_parser = argparse.ArgumentParser(description='Pass in the necessary arguments to predict from the network')

arg_parser.add_argument('path_to_image', help='Path to the image whose prediction is required')
arg_parser.add_argument('checkpoint_directory', help='Path of the model checkpoint')
arg_parser.add_argument('--top_k', action='store', help="Number of top classes, default=5", type=int, default=1)
arg_parser.add_argument('--category_names', action='store', help="Mapping of categories to real names, default=None", default=None)
arg_parser.add_argument('--gpu', action='store_true', default=False, help="To use GPU for prediction or not")

args = arg_parser.parse_args()
print(args)

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

cat_to_name = args.category_names
if cat_to_name is not None:
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

print(str(len(cat_to_name)) + ' classes in categories to real names file')
    
def build_model(arch, n_inputs, n_outputs, n_hidden_layers):
    layers = OrderedDict()
    layers['hidden_1'] = nn.Linear(n_inputs, n_hidden_layers[0])
    layers['relu_1'] = nn.ReLU()
    layers['dropout_1'] = nn.Dropout(0.2)
    for i in range(len(n_hidden_layers)):
        if i != len(n_hidden_layers) - 1:
            layers['hidden_' + str(i+2)] = nn.Linear(n_hidden_layers[i], n_hidden_layers[i+1])
            layers['relu_' + str(i+2)] = nn.ReLU()
            layers['dropout_' + str(i+2)] = nn.Dropout(0.2)
        else:
            continue
    layers['output'] = nn.Linear(n_hidden_layers[-1], n_outputs)
    layers['softmax'] = nn.LogSoftmax(dim=1)
    loaded_classifier = nn.Sequential(layers)
    
    loaded_model = models.vgg16(pretrained=True) if arch == 'vgg16' else models.vgg13(pretrained=True)
    loaded_model.classifier = loaded_classifier
    return loaded_model
    
    
# Assumption: ReLU is being used in the checkpointed model as an activation function, all FC layers have a dropout of p=0.2, and output is a LogSoftMax layer
def load_checkpointed_model(filepath):
    loaded_classifier = torch.load(filepath)
    loaded_model = build_model(loaded_classifier['architecture'], loaded_classifier['input_size'], loaded_classifier['output_size'], loaded_classifier['hidden_layers'])
    loaded_model.load_state_dict(loaded_classifier['state_dict'])
    return loaded_model, loaded_classifier['class_to_index_mapping']

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im.thumbnail((256, 256)) # Resize
    im = im.crop((0, 0, 224, 224)) # Crop
    
    im = np.array(im) # Normalize values
    scaler = 1.0 / (np.max(im) - np.min(im))
    im = im * scaler - np.min(im) * scaler
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std # Standardize according to the network
    im = im.transpose((2, 0, 1)) # Transpose
    ret = torch.Tensor(im)
    return ret

def predict(image_path, model, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    top_p = 0
    top_class = 0
    with torch.no_grad():
        img = process_image(image_path)
        img = img.type(torch.cuda.FloatTensor) if device == 'cuda' else img.type(torch.FloatTensor)
        model = model.to(device)
        img = img.to(device)
        
        model.eval()
        log_ps = model(img.unsqueeze(0))
        ps = torch.exp(log_ps)
        top_probs, top_indices = ps.topk(topk, dim=1)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        index_to_class_mapping = {}
        top_classes = []
        for key, val in loaded_class_to_index_mapping.items():
            index_to_class_mapping[val] = key
        for index in top_indices:
            top_classes.append(index_to_class_mapping[index])
    return top_probs, top_classes
              
loaded_model, loaded_class_to_index_mapping = load_checkpointed_model(args.checkpoint_directory)
print(str(len(loaded_class_to_index_mapping.keys())) + ' classes in class to index mapping')
              
top_p, top_c = predict(args.path_to_image, loaded_model, topk=args.top_k)

classes = []
for c in top_c:
    classes.append(cat_to_name[c])

for i in range(args.top_k):
    print("Class: {} --> Probability: {:.3f}".format(classes[i], top_p[i]))