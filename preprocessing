import pandas as pd 
import pydicom
import os 
import numpy as np  
import matplotlib
import matplotlib.pylab as plt
import cv2

allData = pd.read_csv("list_of_dicom_files.csv") 
Fail=[]
Done=[]
for i in range(len(allData)):
    try:
        ds = pydicom.read_file(allData.iloc[i][0])      
        img = ds.pixel_array.astype(np.float64)        
        if(('RescaleSlope' in ds) and ('RescaleIntercept' in ds)):
            img = (img * ds.RescaleSlope) + ds.RescaleIntercept          
        try:
            img = pydicom.pixel_data_handlers.apply_voi_lut(img, ds) 
        except:                      
            if('WindowCenter' in ds):
                if(type(ds.WindowCenter) == pydicom.multival.MultiValue):
                    window_center = float(ds.WindowCenter[0])
                    window_width = float(ds.WindowWidth[0])
                    lwin = window_center - (window_width / 2.0)
                    rwin = window_center + (window_width / 2.0)
                else:    
                    window_center = float(ds.WindowCenter)
                    window_width = float(ds.WindowWidth)
                    lwin = window_center - (window_width / 2.0)
                    rwin = window_center + (window_width / 2.0)
            else:
                lwin = np.min(img)
                rwin = np.max(img)
                
            img[np.where(img < lwin)] = lwin
            img[np.where(img > rwin)] = rwin                      
        
        if ds[0x0028, 0x0004].value != "MONOCHROME2" : #monochrome1 -> reverted
            norm_img = 1 - (img - np.min(img)) / (np.max(img) - np.min(img)) 
        else: 
            norm_img = (img - np.min(img)) / (np.max(img) - np.min(img))
                
        save_img = np.uint8(norm_img*255)    
        save_img=save_img[0:save_img.shape[1],0:save_img.shape[1]]
        save_img=cv2.resize(save_img, (512, 512)) #512x512
        
        matplotlib.pyplot.imsave('cxr%s.png' %str(i), save_img, cmap='gray') #saving png files
        
        Done.append(allData.iloc[i][0]) 
        
    except:
        Fail.append(allData.iloc[i][0])   
