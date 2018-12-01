import argparse

import torch

def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=True, help='Use GPU if available')
    parser.add_argument('--data_dir', default='flowers', type=str, help='Path to dataset')
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')
    parser.add_argument('--arch', default='densenet121', type=str, help='Model architecture')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--hidden_units', default=[500], type=int, help='Number of hidden units')
    parser.add_argument('--filename', default='./checkpoint.pth', type=str, help='Filename to save checkpoint')
    
    return parser.parse_args()

def parse_predict_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=True, help='Use GPU if available')
    parser.add_argument('--image_path', default='./flowers/test/99/image_07874.jpg', type=str, help='Image to predict')
    parser.add_argument('--mapping', default='./cat_to_name.json', type=str, help='Mapping of categories to real names')
    parser.add_argument('--filename', default='./checkpoint.pth', type=str, help='Filename to save checkpoint')
    
    return parser.parse_args()

def get_device(gpu):
    device = torch.device('cuda:0' if torch.cuda.is_available() and gpu == True else 'cpu')
    
    return device