from __future__ import division
import argparse
import os
import numpy as np
import json

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='GCN train and validate')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--config', dest='config', default='config.json',
                    help='hyperparameter of faster-rcnn in json format')

def load_config(config_path):
    assert(os.path.exists(config_path))
    cfg = json.load(open(config_path, 'r'))
    return cfg

def build_data_loader():
    """
    Build dataloader 
    Args: 
    Return: dataloader for training, validation 
    """
    # train_loader = DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle= True,
    #     num_workers=args.workers, pin_memory=False)
    # val_loader = DataLoader(
    #     val_dataset, batch_size=1, shuffle=False,
    #     num_workers=0, pin_memory=False)   
    # test_loader = DataLoader(
    #     test_loader, batch_size=1, shuffle=False,
    #     num_workers=0, pin_memory=False)     
    # return train_loader, val_loader
def train():
    pass
def validate():
    pass
def main():
    global args
    args = parser.parse_args()
    cfg = load_config(args.config)
    pass
if __name__ == '__main__':
    main()

