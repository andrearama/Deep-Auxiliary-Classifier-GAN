#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:08:56 2018

@author: andrea
"""
import os
import glob
import cv2
import numpy as np
import pickle
import os.path

def load_dataset(image_size, create_new):
    c1 = os.path.isfile("saved_dataset/X_train.pickle")
    c2 = os.path.isfile("saved_dataset/y_train.pickle")
    
    if c1 and c2 and not create_new:
        
        with open('saved_dataset/X_train.pickle', 'rb') as data:
            X = pickle.load(data)

        with open('saved_dataset/y_train.pickle', 'rb') as data:
            y = pickle.load(data)
        
        number_of_classes = max(y)[0] +1
        print("Dataset loaded successfully")
    else:
        X, y, number_of_classes = create_dataset(image_size)
        
    return X, y, number_of_classes
    
    
    
    
def create_dataset(image_size):    
        """
        Create the dataset in the required format.
        It assumes that the images are devided per class in subfolders inside the 
        folder called data
        """
        
        print("Creating the dataset (might take a little)")

        label_ref = "Mapping from folder class to numberic label used: \n\n"
        X = []
        y = []
        folder_list = glob.glob("data/*")
        number_of_classes = len([folder for folder in folder_list if os.path.isdir(folder)])
        # Parse through each folder:
        folder_counter = -1
        for folder_name in folder_list:
            if os.path.isdir(folder_name):
                folder_counter += 1
                label_ref += folder_name+" : "+str(folder_counter )+"\n"
                
                image_list = glob.glob(folder_name+"/*")     
                # Parse through each image in the current folder:
                for image_name in image_list:
                    X, y = add_image(X, y, image_name, image_size, folder_counter)
                    

        #Normalize and format the data:
        X = np.array(X)
        X = (X-127.5)/127.5
        y = np.array(y)
        y = y.reshape(-1,1)
         
        #Shuffle: 
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)        

        with open('saved_dataset/X_train.pickle', 'wb') as output:
            pickle.dump(X, output)

        with open('saved_dataset/y_train.pickle', 'wb') as output:
            pickle.dump(y, output)
        #txt file showing the mapping from folder class to numberic label used
        with open("labels.txt", "w") as text_file:
            print(label_ref , file=text_file)                    
            
        print("""Dataset correctly created and saved. 
              Total number of samples: """+str(X.shape[0]) )
        
        return X, y, number_of_classes
    
    
def add_image(X, y, image_name, image_size, folder_counter, 
              roteate = False, mirror = False):

    img = cv2.imread(image_name)
    if img is not None:
        img = cv2.resize(img, (image_size, image_size), interpolation = cv2.INTER_AREA)
        X.append(img.astype(np.float32))
        y.append(np.uint8(folder_counter))         

        # Roteate 90-180-270 degrees:
        if roteate:
            for i in range(3):
                img = np.rot90(img)
                X.append(img.astype(np.float32))
                y.append(np.uint8(folder_counter)) 
            #restore image to original orientation                
            img = np.rot90(img) 
        
        # Mirror horizontal and vertical
        if mirror:
            img_hor = cv2.flip( img, 0 )
            X.append(img_hor.astype(np.float32))
            y.append(np.uint8(folder_counter)) 

            img_ver = cv2.flip( img, 1 )
            X.append(img_ver.astype(np.float32))
            y.append(np.uint8(folder_counter)) 
            
    else:
        print("Could not load ",image_name,"Is it an image?")    
        
    return X, y