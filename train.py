#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */aipnd-project/train.py

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

import argparse

def get_input_args():
    
    parser = argparse.ArgumentParser(description='Get network arguments')

    parser.add_argument('--data_dir', default='', type=str, help='Path to dataset')
    parser.add_argument('--arch_model', default='vgg19', type=str, help='The model architecture to use')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='The learning rate')
    parser.add_argument('--hidden_units', default=1024, type=int, help='Number of hidden units')
    parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')
    parser.add_argument('--output_size', default=102, type=int, help='The amount of categories of your dataset')
    parser.add_argument('--epochs', default=10, type=int, help='The number of epochs')
    parser.add_argument('--gpu', default=False, action='store_true', help='GPU processing')
    
    return parser.parse_args()


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)


    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)

    image_datasets = [train_data, validation_data, test_data]
    dataloaders = [trainloader, validationloader, testloader]
    
    return trainloader, validationloader, testloader, dataloaders, image_datasets
    
def cnn_config(arch_model, output_size, learning_rate, hidden_units):
    if arch_model == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch_model == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = model.classifier[1].in_features
    elif arch_model == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
    else:
        print("Im sorry but {} is not a valid model. You can choose vgg19, alexnet or densenet121".format(arch_model))
        
    for param in model.parameters():
        param.requires_grad = False    
            
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(input_size, hidden_units)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(hidden_units, output_size)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))
   
        model.classifier = classifier 
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)
        
        return model, criterion, optimizer, input_size, hidden_units

    
def train_model(epochs, model, dataloaders, optimizer, criterion, gpu):
    epochs = epochs
    print_every = 5
    steps = 0
    
    if gpu and torch.cuda.is_available():
        print('Using GPU for Training')
        model.cuda()
    else:
        print('Using CPU for Training')


    running_loss = 0
    accuracy = 0

    start = time.time()
    print('Training data')

    for e in range(epochs):
    
        train_mode = 0
        valid_mode = 1
    
        for mode in [train_mode, valid_mode]:   
            if mode == train_mode:
                model.train()
            else:
                model.eval()
            
            pass_through = 0
        
            for data in dataloaders[mode]:
                pass_through += 1
                inputs, labels = data
                if gpu and torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                output = model.forward(inputs)
                loss = criterion(output, labels)
            
                if mode == train_mode:
                    loss.backward()
                    optimizer.step()                

                running_loss += loss.item()
                ps = torch.exp(output).data
                equality = (labels.data == ps.max(1)[1])
                accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()

            if mode == train_mode:
                print("\nEpoch: {}/{} ".format(e+1, epochs),
                      "\nTraining Loss: {:.4f}  ".format(running_loss/pass_through))
            else:
                print("Validation Loss: {:.4f}  ".format(running_loss/pass_through),
                  "Accuracy: {:.4f}".format(accuracy))

            running_loss = 0

    time_elapsed = time.time() - start
    print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    
def validation_test(model, dataloaders, gpu):
    model.eval()
    accuracy = 0

    if gpu and torch.cuda.is_available():
        print('Using GPU for Training')
        model.cuda()
    else:
        print('Using CPU for Training')
    
    pass_through = 0

    for data in dataloaders[2]:
        pass_through += 1
        inputs, labels = data
        if gpu and torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        

        output = model.forward(inputs)
        ps = torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("Testing Accuracy: {:.4f}".format(accuracy/pass_through))
    
def save_checkpoint(checkpoint, model, image_datasets, arch_model, hidden_units):
    model.class_to_idx = image_datasets[0].class_to_idx

    checkpoint = {'hidden_units': hidden_units,
                  'arch': arch_model,
                  'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')

def main():
    in_args = get_input_args()
    # validationloader, trainloader, testloader = load_data(in_args.data_dir)
    trainloader, validationloader, testloader, dataloaders, image_datasets = load_data(in_args.data_dir)
    
    model, criterion, optimizer, input_size, hidden_units = cnn_config(in_args.arch_model, in_args.output_size, in_args.learning_rate, in_args.hidden_units)
    trained_model = train_model(in_args.epochs, model, dataloaders, optimizer, criterion, in_args.gpu)
    validation = validation_test(model, dataloaders, in_args.gpu)
    save_model = save_checkpoint(in_args.checkpoint, model, image_datasets, in_args.arch_model, in_args.hidden_units)
    pass

if __name__ == '__main__':
    main()



    
    
    
    
        
        
    


    
