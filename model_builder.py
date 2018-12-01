from collections import OrderedDict
import json
import os

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

def get_dataloaders(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])\
            for x in ['train', 'valid', 'test']
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)\
        for x in ['train', 'valid', 'test']
    }
    
    return dataloaders, image_datasets

def get_labelmapping(mapping):
    with open(mapping, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name

def get_model(arch, hidden_units):
    model = getattr(models, arch)(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    if isinstance(model.classifier, nn.Sequential):
        input_features = model.classifier[0].in_features
    else:
        input_features = model.classifier.in_features
        
    hidden_units = [input_features] + hidden_units + [102]
    
    model.classifier = _create_classifier(hidden_units)
    
    return model

def _create_classifier(hidden_units):
    layer_sizes = list(zip(hidden_units[:-1], hidden_units[1:]))
    modules = OrderedDict()

    for count, layer in enumerate(layer_sizes, 1):
        name = 'fc' + str(count)
        modules[name] = nn.Linear(layer[0], layer[1])
        if count == len(layer_sizes):
            modules['output'] = nn.LogSoftmax(dim=1)
        else:
            relu = 'relu' + str(count)
            modules[relu] = nn.ReLU()
            dropout = 'dropout' + str(count)
            modules[dropout] = nn.Dropout(p=0.5)
    
    return nn.Sequential(modules)

def validation(model, dataloader, criterion, device):
    loss = 0
    accuracy = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return loss, accuracy

def train_model(model, epochs, trainloader, validloader, criterion, optimizer, device):
    steps = 0
    running_loss = 0
    print_every = 40
    
    model.to(device)
    
    for e in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                # Make sure training is back on
                model.train()

def test_model(model, testloader, criterion, device):
    model.eval()
    with torch.no_grad():
        test_loss, accuracy = validation(model, testloader, criterion, device)

    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
    
def save_checkpoint(arch, hidden_units, state_dict, class_to_idx, filename):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'state_dict': state_dict,
        'class_to_idx': class_to_idx
    }
    
    torch.save(checkpoint, filename)
    
def load_model(filename):
    checkpoint = torch.load(filename)
    model = get_model(checkpoint['arch'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return preprocess(image)

def predict(image_path, model, topk, device):
    image = process_image(image_path)
    image.unsqueeze_(0)
    
    model.to(device)
    image = image.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
        ps, indices = torch.exp(output).cpu().topk(topk)
    
    ps = ps.data.numpy().squeeze()
    indices = indices.data.numpy().squeeze()
    
    idx_to_class = {y:x for x, y in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]
       
    return ps, classes
    