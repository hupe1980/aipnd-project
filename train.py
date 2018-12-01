#!/usr/bin/env python3
import torch
from torch import optim
from torch import nn

import model_builder
import utils

def main():
    args = utils.parse_train_args()
    
    dataloaders, datasets = model_builder.get_dataloaders(args.data_dir)
    model = model_builder.get_model(args.arch, args.hidden_units)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
    
    model_builder.train_model(model, args.epochs, dataloaders['train'], dataloaders['valid'], criterion, optimizer, device)
    model_builder.test_model(model, dataloaders['test'], criterion, device)
    
    model_builder.save_checkpoint(args.arch, args.hidden_units, model.state_dict(), datasets['train'].class_to_idx, args.filename)

if __name__ == "__main__":
    main()