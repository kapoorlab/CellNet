import csv
import numpy as np
from tifffile import imread, imwrite 
import pandas as pd
import os
import glob
from skimage.measure import regionprops
from skimage import measure
from scipy import spatial 
from pathlib import Path


def ImageLabelDataSet(ImageDir, SegImageDir, CSVDir,SaveDir, StaticName, StaticLabel, CSVNameDiff,crop_size, gridX = 1, gridY = 1, offset = 0):
    
    
            Raw_path = os.path.join(ImageDir, '*tif')
            Seg_path = os.path.join(SegImageDir, '*tif')
            Csv_path = os.path.join(CSVDir, '*csv')
            filesRaw = glob.glob(Raw_path)
            filesRaw.sort
            filesSeg = glob.glob(Seg_path)
            filesSeg.sort
            filesCsv = glob.glob(Csv_path)
            filesCsv.sort
            Path(SaveDir).mkdir(exist_ok=True)
            TotalCategories = len(StaticName)
            count = 0
            for csvfname in filesCsv:
              print(csvfname)
              CsvName =  os.path.basename(os.path.splitext(csvfname)[0])
            
              for fname in filesRaw:
                  
                 Name = os.path.basename(os.path.splitext(fname)[0])   
                 for Segfname in filesSeg:
                      
                      SegName = os.path.basename(os.path.splitext(Segfname)[0])
                        
                      if Name == SegName:
                          
                          
                         image = imread(fname)
                         segimage = imread(Segfname)
                         for i in  range(0, len(StaticName)):
                             Eventname = StaticName[i]
                             trainlabel = StaticLabel[i]
                             if CsvName == CSVNameDiff + Name + Eventname:
                                            dataset = pd.read_csv(csvfname)
                                            time = dataset[dataset.keys()[0]][1:]
                                            y = dataset[dataset.keys()[1]][1:]
                                            x = dataset[dataset.keys()[2]][1:]                        
                                            #Categories + XYHW + Confidence 
                                            for t in range(1, len(time)):
                                               ImageMaker(time[t], y[t], x[t], image, segimage, crop_size, gridX, gridY, offset, TotalCategories, trainlabel, Name + str(count), SaveDir)    
                                               count = count + 1
    



def  ImageMaker(time, y, x, image, segimage, crop_size, gridX, gridY, offset, TotalCategories, trainlabel, name, save_dir):

       sizeX, sizeY = crop_size
       
       ImagesizeX = sizeX * gridX
       ImagesizeY = sizeY * gridY
       
       shiftNone = [0,0]
       if offset > 0:
         shiftLX = [-1.0 * offset, 0] 
         shiftRX = [offset, 0]
         shiftLXY = [-1.0 * offset, -1.0 * offset]
         shiftRXY = [offset, -1.0 * offset]
         shiftDLXY = [-1.0 * offset, offset]
         shiftDRXY = [offset, offset]
         shiftUY = [0, -1.0 * offset]
         shiftDY = [0, offset]
         AllShifts = [shiftNone, shiftLX, shiftRX,shiftLXY,shiftRXY,shiftDLXY,shiftDRXY,shiftUY,shiftDY]

       else:
           
          AllShifts = [shiftNone]

       
       try:
               currentsegimage = segimage[time,:].astype('uint16')
               for shift in AllShifts:
                   
                        defaultX = int(x + shift[0])  
                        defaultY = int(y + shift[1])
                        #Get the closest centroid to the clicked point
                        properties = measure.regionprops(currentsegimage, currentsegimage)
                        TwoDCoordinates = [(prop.centroid[0], prop.centroid[1]) for prop in properties]
                        TwoDtree = spatial.cKDTree(TwoDCoordinates)
                        TwoDLocation = (defaultY,defaultX)
                        closestpoint = TwoDtree.query(TwoDLocation)
                        for prop in properties:
                                           
                                           
                                    if int(prop.centroid[0]) == int(TwoDCoordinates[closestpoint[1]][0]) and int(prop.centroid[1]) == int(TwoDCoordinates[closestpoint[1]][1]):
                                                        minr, minc, maxr, maxc = prop.bbox
                                                        center = prop.centroid
                                                        height =  abs(maxc - minc)
                                                        width =  abs(maxr - minr)
                                                        break
                        
                        Label = np.zeros([TotalCategories + 5])
                        Label[trainlabel] = 1
                        
                        if x + shift[0]> sizeX/2 and y + shift[1] > sizeY/2 and x + shift[0] < image.shape[2] and y + shift[1] < image.shape[1]:
                                    crop_Xminus = x + shift[0] - int(ImagesizeX/2)
                                    crop_Xplus = x + shift[0] + int(ImagesizeX/2)
                                    crop_Yminus = y + shift[1] - int(ImagesizeY/2)
                                    crop_Yplus = y + shift[1] + int(ImagesizeY/2)
              
                                    region =(slice(int(time - 1),int(time)),slice(int(crop_Yminus), int(crop_Yplus)),
                                           slice(int(crop_Xminus), int(crop_Xplus)))
                                    crop_image = image[region]      
                                    Label[TotalCategories] =  (center[1] - crop_Xminus)/sizeX
                                    Label[TotalCategories + 1] = (center[0] -  crop_Yminus)/sizeY
                                    Label[TotalCategories + 2] = height/ImagesizeY
                                    Label[TotalCategories + 3] = width/ImagesizeX
                                   
                                    #Object confidence is 0 for background label else it is 1
                                    if trainlabel > 0:
                                      Label[TotalCategories + 4] = 1
                                    else:
                                      Label[TotalCategories + 4] = 0 
                                  
                                    if(crop_image.shape[1]== ImagesizeY and crop_image.shape[2]== ImagesizeX):
                                             imwrite((save_dir + '/' + name + '.tif'  ) , crop_image.astype('float32'))  
                                   
                                    writer = csv.writer(open(save_dir + '/' + (name) + ".csv", "w"))
                                    for l in Label : writer.writerow ([l])

       

       except:
          
          pass

