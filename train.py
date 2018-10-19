#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:55:19 2018

@author: andrea
"""
import argparse
from model import Acgan


def unpack_flags(args):
    
    global epochs, batch_size, img_size, num_classes
    global create_new, flip_img, roteate_img
    
    epochs = args.epochs
    batch_size = args.batch_size
    img_size = args.img_size
    num_classes = args.num_classes

    if args.flip_img:
        flip_img = True
    else:
        flip_img = False

    if args.roteate_img:
        roteate_img = True
    else:
        roteate_img = False
        
    if args.create_new or roteate_img or flip_img:
        create_new = True
    else:
        create_new = False
    
    if img_size % 8 != 0:
        raise ValueError("The size of the image has to be multiple of 8")
    
    if num_classes > 10:
        print("""Warning: It is not recommended to use a single model for so many
              classes. To avoid problems like mode collapse and an overall
              difficulty in the training, you might want to consider using multiple
              ACGAN for subsets of classes, as done also in the original paper""")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=14000,type = int, help='Number of training epoch')
    parser.add_argument('--batch_size', default=32,type = int, help='Number of images per batch')
    parser.add_argument('--img_size', default=28, type = int, help='size of the image (assumed squared)')
    parser.add_argument('--num_classes', default=10, type = int, help='number of classes')
    parser.add_argument('--create_new', action='store_true', help='create new dataset regardless if there is already one')
    parser.add_argument('--flip_img', action='store_true', help='create new dataset regardless if there is already one')
    parser.add_argument('--roteate_img', action='store_true', help='create new dataset regardless if there is already one')
    
    args = parser.parse_args()
    unpack_flags(args)   
    
    acgan = Acgan(img_size, num_classes, create_new)
    acgan.train(epochs, batch_size, flip_img, roteate_img, sample_interval=50)