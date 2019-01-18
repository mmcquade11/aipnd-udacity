#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */aipnd-project/predict.py

import time
import json
import copy
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns
import numpy as np
from PIL import Image
from collections import OrderedDict
import torch.nn.functional as F
import torch
from torch import nn, optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import torchvision.models as models
import train 

import argparse

def get_input_args():
    
    parser = argparse.ArgumentParser(description='Get arguments for predict')

    parser.add_argument('--data_dir', default='', type=str, help='Path to dataset')
    parser.add_argument('--image', type=str, help='An image to load for processing')
    parser.add_argument('--top_k', default=5, type=int, help='top_k results')
    parser.add_argument('--category_names', default='', type=str, help='file with category names')
    parser.add_argument('--checkpoint', default='checkpoint.pth', type=str, help='checkpoint file to load')
    parser.add_argument('--gpu', default=False, action='store_true', help='GPU processing')
    
    return parser.parse_args()

def load_checkpoint(checkpoint):
    trained_model = torch.load('checkpoint.pth')
    arch = trained_model['arch']
    class_idx = trained_model['class_to_idx']

    if arch == 'vgg19':
        load_model = models.vgg19(pretrained=True)
    elif arch == 'alexnet':
        load_model = models.alexnet(pretrained=True)
    elif arch == 'densenet121':
        load_model = models.densenet121(pretrained=True)
    else:
        print('{} architecture not recognized. Supported args: \'vgg\', \'alexnet\', or \'densenet\''.format(arch))
        
    for param in load_model.parameters():
        param.requires_grad = False
    
    load_model.classifier = trained_model['classifier']
    load_model.load_state_dict(trained_model['state_dict'])
    
    return load_model, arch, class_idx

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image

def predict(image, model, top_k, gpu, category_names, arch, class_idx):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model. Returns top_k classes
    and probabilities. If name json file is passed, it will convert classes to actual names.
    '''
    image = Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0)
   
    image = Variable(image)
    
    if gpu and torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
        print('GPU PROCESSING')
    else:
        print('CPU PROCESSING')
    with torch.no_grad():
        out = model.forward(image)
        results = torch.exp(out).data.topk(top_k)
    classes = np.array(results[1][0], dtype=np.int)
    probs = Variable(results[0][0]).data
    
    if len(category_names) > 0:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        #Creates a dictionary of loaded names based on class_ids from model
        mapped_names = {}
        for k in class_idx:
            mapped_names[cat_to_name[k]] = class_idx[k]
        #invert dictionary to accept prediction class output
        mapped_names = {v:k for k,v in mapped_names.items()}
        
        classes = [mapped_names[x] for x in classes]
        probs = list(probs)
    else:
        #Invert class_idx from model to accept prediction output as key search
        class_idx = {v:k for k,v in class_idx.items()}
        classes = [class_idx[x] for x in classes]
        probs = list(probs)
    return classes, probs

def print_predict(classes, probs):
    
    predictions = list(zip(classes, probs))
    for i in range(len(predictions)):
        print('{} : {:.3%}'.format(predictions[i][0], predictions[i][1]))
    pass


def main():
    in_args = get_input_args()
    norm_image = process_image(in_args.image)
    model, arch, class_idx = load_checkpoint(in_args.checkpoint)
    classes, probs = predict(norm_image, model, in_args.top_k, in_args.gpu, in_args.category_names, arch, class_idx)
    print_predict(classes, probs)
    pass
if __name__ == '__main__':
    main()                                    
                                  
