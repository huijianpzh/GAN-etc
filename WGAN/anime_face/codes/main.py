#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 13:35:18 2018

@author: huijian
"""

# my lib
from networks.dcgan import (Generator, Discriminator)
from trainer import train_wgan_gp
from dataio import Faces
from dataio import Nptranspose

# torch lib
import torch
import torch.utils as utils
import torchvision.transforms as transforms

# other libs
import numpy as np
from skimage import io

if __name__ == "__main__":
    
    # some parameters
    file_path = "../model"
    batch_size = 256
    epoch =500
    #cuda = False
    cuda = torch.cuda.is_available()
    
    z_dim = 100
    in_channels=3
    out_channels=3
    base = 64
    
    
    # prepare data
    data_dir = "../faces"
    composed = transforms.Compose([Nptranspose()])
    faces = Faces(data_dir = data_dir, data_transforms=composed)
    
    
    name = "dcgan_"
    if False:
        gen =  Generator(z_dim, out_channels, base)
        disc = Discriminator(in_channels, base)
        trainer = train_wgan_gp(disc, gen, z_dim, file_path, cuda)
    else:
        trainer = train_wgan_gp(None, None, z_dim, file_path, cuda)
        trainer.restore_model(name)
        
    if False:
        trainer.train_model(faces,epoch,batch_size)
    
    if True:
        name = "../test.png"
        result = trainer.generate_image(file_name=name)
        
