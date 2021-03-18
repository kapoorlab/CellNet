#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:13:01 2020

@author: Varun Kapoor
"""

import numpy as np
from utils import helpers
from keras import callbacks
import os
from models import nets
from keras import backend as K
#from IPython.display import clear_output
from keras import optimizers
from pathlib import Path



class CellNet(object):
    

    """
    Parameters
    ----------
    config : config is an object containing information about the network parameters, this object is also saved as a json file
    
    TrainDirectory : Specify the location of the directory containing the training data with movies and labels
    
    Categories_Name : List of class names and labels
    
    box_vector : Number of components other than the categories that the network has to output to perform localization
    
    model_dir : Directory location where trained model weights are to be read or written from
    
    model_name : The h5 file of CNN + LSTM + Dense Neural Network to be used for training
    
    model_weights : If re-training model_weights = model_dir + model_name else None as default
    
    show : If true the results of trainng will be displayed every epoch on a randomly chosen image from the validation set along with the loss and accuracy plot. THis option is set true
    if you are using jupyter notebooks but if executing this program from a py file this parameter is set to false.
    
    
    
    
    """
    
    
    def __init__(self, staticconfig, TrainDirectory, KeyCatagories, KeyCord, model_dir, model_name,  show = False ):

        self.TrainDirectory = TrainDirectory
        self.model_dir = model_dir
        self.model_name = model_name
        self.KeyCatagories = KeyCatagories
        self.box_vector = len(KeyCord)
        self.model_weights = None
        self.show = show
        self.simple = staticconfig.simple
        self.categories = len(KeyCatagories)
        self.depth = staticconfig.depth
        self.start_kernel = staticconfig.start_kernel
        self.mid_kernel = staticconfig.mid_kernel
        self.learning_rate = staticconfig.learning_rate
        self.epochs = staticconfig.epochs
        self.residual = staticconfig.residual
        self.startfilter = staticconfig.startfilter
        self.batch_size = staticconfig.batch_size
        self.multievent = staticconfig.multievent
        
        self.nboxes = staticconfig.nboxes
        self.gridX = staticconfig.gridX
        self.gridY = staticconfig.gridY
        self.lambdacord = staticconfig.lambdacord
        self.last_activation = None
        self.entropy = None
        self.X = None
        self.Y = None
        self.X_val = None
        self.Y_val = None
        self.Trainingmodel = None

    def loadData(self):
        
        (X,Y), (X_val,Y_val) = helpers.load_full_training_data(self.TrainDirectory, self.categories, self.nboxes)

        self.X = X
        self.Y = Y
        self.X_val = X_val
        self.Y_val = Y_val
          

              
    def TrainModel(self):
        
        # input shape is  H W C
        input_shape = (self.X.shape[1], self.X.shape[2], self.X.shape[3])
        
        Path(self.model_dir).mkdir(exist_ok=True)
        
        
        
        if self.residual:
            model_keras = nets.resnet_v2
        else:
            model_keras = nets.seqnet_v2
        
        if self.multievent == True:
           self.last_activation = 'sigmoid'
           self.entropy = 'binary'
           
           
        if self.multievent == False:
           self.last_activation = 'softmax'              
           self.entropy = 'notbinary' 
           
        model_weights = self.model_dir + self.model_name
        if os.path.exists(model_weights):
        
            self.model_weights = model_weights
            print('loading weights')
        else:
           
            self.model_weights = None    
           
        
        self.Trainingmodel = model_keras(input_shape, self.categories, box_vector = self.box_vector , depth = self.depth, start_kernel = self.start_kernel, mid_kernel = self.mid_kernel, startfilter = self.startfilter, 
                                         gridX = self.gridX, gridY = self.gridY, nboxes = self.nboxes,  last_activation = self.last_activation, input_weights  =  self.model_weights)
        
            
        sgd = optimizers.SGD(lr=self.learning_rate, momentum = 0.99, decay=1e-6, nesterov = True)
        
        self.Trainingmodel.compile(optimizer=sgd, loss = static_yolo_loss(self.categories, self.gridX, self.gridY, self.nboxes, self.box_vector, self.lambdacord, self.entropy), metrics=['accuracy'])
        self.Trainingmodel.summary()
        print('Training Model:', model_keras)
        
        #Keras callbacks
        lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
        hrate = callbacks.History()
        srate = callbacks.ModelCheckpoint(self.model_dir + self.model_name, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        prate = plotters.PlotStaticHistory(self.Trainingmodel, self.X_val, self.Y_val, self.KeyCatagories, self.gridX, self.gridY, plot = self.show, nboxes = self.nboxes)
        
        
        #Train the model and save as a h5 file
        self.Trainingmodel.fit(self.X,self.Y, batch_size = self.batch_size, epochs = self.epochs, validation_data=(self.X_val, self.Y_val), shuffle = True, callbacks = [lrate,hrate,srate,prate])
     
        # Removes the old model to be replaced with new model, if old one exists
        if os.path.exists(self.model_dir + self.model_name ):

           os.remove(self.model_dir + self.model_name )
        
        self.Trainingmodel.save(self.model_dir + self.model_name )
        
    




def static_yolo_loss(categories, gridX, gridY, nboxes, box_vector, lambdacord, entropy):
    
    def loss(y_true, y_pred):
        
       
        grid = np.array([ [[float(x),float(y)]]*nboxes   for y in range(gridY) for x in range(gridX)])
        
        y_true_class = y_true[...,0:categories]
        y_pred_class = y_pred[...,0:categories]
        
        
        pred_boxes = K.reshape(y_pred[...,categories:], (-1, gridY * gridX, nboxes, box_vector))
        true_boxes = K.reshape(y_true[...,categories:], (-1, gridY * gridX, nboxes, box_vector))
        
        y_pred_xy = pred_boxes[...,0:2] +  (grid)
        y_true_xy = true_boxes[...,0:2]
        
        y_pred_hw = pred_boxes[...,2:4]
        y_true_hw = true_boxes[...,2:4]
        
        y_pred_conf = pred_boxes[...,4]
        y_true_conf = true_boxes[...,4]
        
        
        if entropy == 'notbinary':
            class_loss = K.mean(K.categorical_crossentropy(y_true_class, y_pred_class), axis=-1)
        if entropy == 'binary':
            class_loss = K.mean(K.binary_crossentropy(y_true_class, y_pred_class), axis=-1)
        else:
            class_loss = K.mean(K.categorical_crossentropy(y_true_class, y_pred_class), axis=-1)
        xy_loss = K.sum(K.sum(K.square(y_true_xy - y_pred_xy), axis = -1) * y_true_conf, axis = -1)
        hw_loss = K.sum(K.sum(K.square(K.sqrt(y_true_hw) - K.sqrt(y_pred_hw)), axis=-1)*y_true_conf, axis=-1)
      
        #IOU computation for increasing localization accuracy
       
        intersect_wh = K.maximum(K.zeros_like(y_pred_hw), (y_pred_hw + y_true_hw)/2 - K.square(y_pred_xy - y_true_xy) )
        intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
        true_area = y_true_hw[...,0] * y_true_hw[...,1]
        pred_area = y_pred_hw[...,0] * y_pred_hw[...,1]
        union_area = pred_area + true_area - intersect_area
        iou = intersect_area / union_area
        conf_loss = K.sum(K.square(y_true_conf*iou - y_pred_conf), axis=-1)

        combinedloss =  class_loss + lambdacord * ( xy_loss + hw_loss ) + conf_loss
        
        return combinedloss
    
    
    return loss

    
