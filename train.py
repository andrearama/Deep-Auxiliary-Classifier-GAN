#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:55:19 2018

@author: andrea
"""
from model import Acgan
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=14000,type = int, help='Number of training epoch')
    parser.add_argument('--batch_size', default=32,type = int, help='Number of images per batch')
    parser.add_argument('--img_size', default=28, type = int, help='size of the image (assumed squared)')
    parser.add_argument('--num_classes', default=10, type = int, help='number of classes')
    parser.add_argument('--create_new', default=False, type = bool, help='create new dataset regardless if there is already one')
    
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    img_size = args.img_size
    num_classes = args.num_classes
    create_new = args.create_new
    
    if img_size % 4 != 0:
        raise ValueError("The size of the image has to be multiple of 4")
    
    if num_classes > 10:
        print("""Warning: It is not recommended to use a single model for so many
              classes. To avoid problems like mode collapse and an overall
              difficulty in the training, you might want to consider using multiple
              ACGAN for subsets of classes, as done also in the original paper""")
        
    acgan = Acgan(img_size, num_classes, create_new)
    acgan.train(epochs, batch_size, sample_interval=50)