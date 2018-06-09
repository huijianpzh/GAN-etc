#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 23:03:59 2018

@author: huijian
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from skimage import io
from skimage import transform

import torch.utils as utils
import torchvision.transforms as transforms


class Faces(utils.data.Dataset):
    def __init__(self,data_dir,data_transforms=None):
        
        self.data_dir = data_dir
        self.transforms = data_transforms
        
        self.data = []
        
        files = os.listdir(self.data_dir)
        for item in files:
            if item.endswith(".jpg"):
                 self.data.append(item.split(".jpg")[0])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        
        file_path = self.data_dir + "/" + self.data[idx] + ".jpg"
        
        img = io.imread(file_path)
        img = img.astype(np.float)/(255*0.5) - 1 # 0~255 -> -1~1
        
        sample={}
        sample["image"] = img
        
        if self.transforms:
            sample = self.transforms(sample)
        
        return sample

class Nptranspose(object):
    def __call__(self,sample):
        img = sample["image"]
        sample["image"] = img.transpose(2,0,1)
        return sample


if __name__=="__main__":
    print("A example of dataset!")
    
    data_dir = "../faces"
    
    faces = Faces(data_dir = data_dir)
    example_1 = faces[0]["image"]
    
    batch_size=64
    print("faces size : {s}".format(s=len(faces)))
    print("batch_size : {bs}".format(bs=batch_size))
    d_every = 1
    g_every = 5
    
    data_loader = utils.data.DataLoader(dataset=faces,batch_size=batch_size,shuffle=True)    
    # simulation the  iteration in training
    epochs = range(10)
    for e in iter(epochs): # epoch
        for i,image, in tqdm.tqdm(enumerate(data_loader)):
            pass