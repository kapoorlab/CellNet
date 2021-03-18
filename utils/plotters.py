import keras
import matplotlib.pyplot as plt
from matplotlib import cm
import random
"""
@author: Varun Kapoor
"""

class PlotHistory(keras.callbacks.Callback):
    
    
    def __init__(self, Trainingmodel, X, Y, KeyCatagories, gridX, gridY, plot = False, nboxes = 1):
       self.Trainingmodel = Trainingmodel 
       self.X = X
       self.Y = Y
       self.plot = plot
       self.gridX = gridX
       self.gridY = gridY
       self.nboxes = nboxes
      
       self.KeyCatagories = KeyCatagories
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        if self.plot:
         self.fig = plt.figure()
        
        self.logs = []
       
    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        
        self.i += 1
        if self.plot:
         f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        
        
         ax1.set_yscale('log')
         ax1.plot(self.x, self.losses, label="loss")
         ax1.plot(self.x, self.val_losses, label="val_loss")
         ax1.legend()
        
         ax2.plot(self.x, self.acc, label="acc")
         ax2.plot(self.x, self.val_acc, label="val_acc")
         ax2.legend()
         plt.show()
         #clear_output(True)
        idx = random.randint(1,self.X.shape[0] - 1)
        Printpredict(idx,self.Trainingmodel, self.X, self.Y, self.KeyCatagories, self.gridX, self.gridY, plot = self.plot, nboxes = self.nboxes)
        
def Printpredict(idx, model, data, Truelabel, KeyCatagories, gridX, gridY, plot = False, nboxes = 1):

    Image = data[idx]
    Truelabel = Truelabel[idx]
    prediction = model.predict(data)
   
    cols = 5
    
    if plot:  
          import matplotlib.pyplot as plt  
          fig, ax = plt.subplots(1,data.shape[1],figsize=(5*cols,5))
          fig.figsize=(20,10)
    # The prediction vector is (1, categories + box_vector) dimensional, input data is (N T H W C) C is 1 in our case
    for j in range(0,data.shape[1]):
            
            img = Image[j,:,:,0]
            if plot:
              ax[j].imshow(img, cm.Spectral)
              
    for (Name, Label) in KeyCatagories.items():
        
             print('Top predictions : ' , Name, 'Probability', ':' , prediction[0,:,:, Label])
                   
             for b in range(nboxes): 
                             print('Box: ' , b)  
                             print('X Y T H W Confidence Angle',prediction[0,:,:,len(KeyCatagories) + b: len(KeyCatagories) + b + 7])
                       
            
    print('True Label : ', Truelabel)

    if plot:
              plt.show()     
        

class PlotStaticHistory(keras.callbacks.Callback):
    
    
    def __init__(self, Trainingmodel, X, Y, KeyCatagories, gridX, gridY, plot = False, nboxes = 1):
       self.Trainingmodel = Trainingmodel 
       self.X = X
       self.Y = Y
       self.gridX = gridX
       self.gridY = gridY
       self.plot = plot
       self.nboxes = nboxes
      
       self.KeyCatagories = KeyCatagories
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        if self.plot:
         self.fig = plt.figure()
        
        self.logs = []
       
    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        
        self.i += 1
        if self.plot:
         f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        
        
         ax1.set_yscale('log')
         ax1.plot(self.x, self.losses, label="loss")
         ax1.plot(self.x, self.val_losses, label="val_loss")
         ax1.legend()
        
         ax2.plot(self.x, self.acc, label="acc")
         ax2.plot(self.x, self.val_acc, label="val_acc")
         ax2.legend()
         plt.show()
         #clear_output(True)
        idx = random.randint(1,self.X.shape[0] - 1)
        PrintStaticpredict(idx,self.Trainingmodel, self.X, self.Y, self.KeyCatagories, self.gridX, self.gridY, plot = self.plot, nboxes = self.nboxes)
        
def PrintStaticpredict(idx, model, data, Truelabel, KeyCatagories, gridX, gridY, plot = False, nboxes = 1):

    Image = data[idx]
    Truelabel = Truelabel[idx]
    prediction = model.predict(data)
   
    # The prediction vector is (1, categories + box_vector) dimensional, input data is (N H W C) C is 1 in our case
            
    img = Image[:,:,0]
    if plot:
        plt.imshow(img, cm.Spectral)
     
    for (Name, Label) in KeyCatagories.items():
        
             print('Top predictions : ' , Name, 'Probability', ':' , prediction[0,:,:, Label])
                   
             for b in range(nboxes): 
                             print('Box: ' , b)  
                             print('X Y T H W Confidence Angle',prediction[0,:,:,len(KeyCatagories) + b: len(KeyCatagories) + b + 5])
                       
            
    print('True Label : ', Truelabel)

    if plot:
              plt.show()          
