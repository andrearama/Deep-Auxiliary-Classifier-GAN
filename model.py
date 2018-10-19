#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:15:51 2018

@author: Andrea
"""
from __future__ import print_function, division

from utils import load_dataset
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import cv2
import numpy as np

class Acgan():
    def __init__(self,img_size, num_classes, create_new):
        # Input shape
        self.img_size = img_size #images are squared
        self.channels = 3
        self.img_shape = (self.img_size, self.img_size, self.channels)
        self.num_classes = num_classes
        self.latent_dim = 100
        self.create_new = create_new
        self.gen_history = []
        self.label_history = []

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        label_tensor = Input(shape=(self.img_size/2, self.img_size/2, 
                                    self.num_classes), dtype='float32')
        img = self.generator([noise, label, label_tensor])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label, label_tensor], [valid, target_label])
        self.combined.compile(loss=losses,
            optimizer=optimizer)

    def build_generator(self):
        d1 = int(self.img_size / 8)
        
        model = Sequential()        
        #first block (Dense)
        model.add(Dense(256 * d1 * d1, input_dim=self.latent_dim, kernel_initializer = 'he_normal'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(rate = 0.4) )        
        model.add(LeakyReLU(alpha=0.2))  
        model.add(Reshape((d1, d1, 256)))                                #size:(d1,d1,256)
        #second block (Convolutional)
        model.add(UpSampling2D(interpolation='bilinear'))                #size: (2*d1,2*d1,256)
        model.add(Conv2D(256, kernel_size=3, padding="same", kernel_initializer = 'he_normal'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(rate = 0.4) )                
        model.add(LeakyReLU(alpha=0.2))               
        #third block (Convolutional and concatenation)          
        model.add(UpSampling2D(interpolation='bilinear'))                #size: (4*d1,4*d1,256)

                      
        model2 = Sequential()                                            #size: (4*d1,4*d1,256+num_classes)
        model2.add(Conv2D(128, kernel_size=3, padding="same",            #size: (4*d1,4*d1,128)
                          input_shape=(4*d1, 4*d1, 256+self.num_classes), kernel_initializer = 'he_normal' ))           
        model2.add(BatchNormalization(momentum=0.8))
        model2.add(LeakyReLU(alpha=0.2))                            
        #fourth block (Convolutional)
        model2.add(UpSampling2D(interpolation='bilinear'))               #size: (img_size,img_size,128)
        model2.add(Conv2D(64, kernel_size=3, padding="same",
                          kernel_initializer = 'he_normal'))             #size: (img_size,img_size,64)
        model2.add(BatchNormalization(momentum=0.8))
        model2.add(LeakyReLU(alpha=0.2))
        #Convolutional layer + activation
        model2.add(Conv2D(self.channels, kernel_size=3, padding='same')) #size: (img_size,img_size,3)
        model2.add(Activation("tanh"))

        model2.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_tensor = Input(shape=(4*d1,4*d1,self.num_classes), dtype='float32')

        label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))
        model_input = multiply([noise, label_embedding])
        r = model(model_input)
        merged = concatenate([r, label_tensor])        

        img = model2(merged)

        return Model([noise, label,label_tensor], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs, replay = True, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train, y_train,number_of_classes = load_dataset(self.img_size,self.create_new)

        #Check the number of classes is right
        if number_of_classes != self.num_classes: 
            raise ValueError("The number of classes found is "+ str(number_of_classes) +
                             " but the number of classes specified is "+ str(self.num_classes)+
                             "\n Maybe there was some empty folder?")


        # Adversarial ground truths
        valid_o = np.ones((batch_size, 1))
        fake_o = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Label smoothing:
            valid = self.label_smoothing(valid_o)
            fake = self.label_smoothing(fake_o)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = np.random.randint(0, self.num_classes, (batch_size, 1))
            label_tensor = self.get_label_tensor(sampled_labels, batch_size)
            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels,label_tensor])            
            # Replay:
            if replay:
                if epoch> 100 and epoch % 10:
                    self.gen_history.append(gen_imgs[0])
                    self.label_history.append(sampled_labels[0])
                if epoch> 200: 
                    gen_imgs, sampled_labels = self.add_replays(gen_imgs, sampled_labels)
                    label_tensor = self.get_label_tensor(sampled_labels, batch_size)
                    
                    
            # Image labels. 0-9 
            img_labels = y_train[idx]

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels, label_tensor], [valid, sampled_labels])

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_model()
                self.sample_images(epoch, batch_size)

    def sample_images(self, epoch, batch_size, class_img = 0):  #TODO: Change 4 batch

        noise = np.random.normal(0, 1, (1, 100))
        sampled_labels = np.array([class_img])
        label_tensor = self.get_label_tensor(sampled_labels, 1)
        gen_imgs = self.generator.predict([noise, sampled_labels, label_tensor])

        # Rescale image to 0 - 255
        gen_imgs = 255 * (0.5 * gen_imgs + 0.5) 
        gen_imgs = gen_imgs.astype(np.int64)

        cv2.imwrite("images/"+str(epoch)+".jpg",gen_imgs[0] )


    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


    def label_smoothing(self, vector, max_dev = 0.2):
        d = max_dev * np.random.rand(vector.shape[0],vector.shape[1])
        if vector[0][0] == 0:
            return vector + d
        else:
            return vector - d
        
        
    def add_replays(self, gen_imgs, sampled_labels, proportion = 0.3) :
        """
        Substitute randomly a portion of the newly generated images with some
        older (generated) ones
        """
        n = int(gen_imgs.shape[0] * proportion)
        n = min(n, len(self.label_history) )
        idx_gen = np.random.randint(0, gen_imgs.shape[0], n)
        idx_hist= np.random.randint(0, len(self.gen_history), n)
        for i_g, i_h in zip(idx_gen, idx_hist) :
            gen_imgs[i_g] =  self.gen_history[i_h]
            sampled_labels[i_g] = self.label_history[i_h] 
        
        return gen_imgs, sampled_labels
    
    
    def get_label_tensor(self, label, batch_size):
        shape =(batch_size, int(self.img_size/2), int(self.img_size/2), self.num_classes)
        t = np.zeros(shape= shape, dtype='float32')
        for i in range(batch_size):
            idx = int( label[i])
            t[i,:,:,idx] = 1

        return t