#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 23:41:20 2018

@author: huijian
"""

import torch
import torch.nn as nn

# this file contains the definition of Generator and Discriminator of DCGAN

class Generator(nn.Module):
    def __init__(self, z_dim=100, out_channels=3, base=64):
        super(Generator,self).__init__()
        # 100->1024
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=z_dim,out_channels=base*16,
                                                      kernel_size=4,stride=1,padding=0,bias=False),
                                   nn.BatchNorm2d(num_features=base*16),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(in_channels=base*16,out_channels=base*8,
                                                      kernel_size=4,stride=2,padding=1,bias=False),
                                   nn.BatchNorm2d(num_features=base*8),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.ConvTranspose2d(in_channels=base*8,out_channels=base*4,
                                                      kernel_size=4,stride=2,padding=1,bias=False),
                                   nn.BatchNorm2d(num_features=base*4),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.ConvTranspose2d(in_channels=base*4,out_channels=base*2,
                                                      kernel_size=4,stride=2,padding=1,bias=False),
                                   nn.BatchNorm2d(num_features=base*2),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(in_channels=base*2,out_channels=out_channels,
                                                      kernel_size=5,stride=3,padding=1,bias=False),
                                   #nn.BatchNorm2d(num_features=out_channels),
                                   nn.Tanh()) # -1~1
    def forward(self,x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        return x

class Discriminator(nn.Module):
    def __init__(self,in_channels=3,base=64):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=base,
                                             kernel_size=5,stride=3,padding=1,bias=False),
                                   #nn.BatchNorm2d(num_features=base),
                                   nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=base,out_channels=2*base,
                                             kernel_size=4,stride=2,padding=1,bias=False),
                                   nn.BatchNorm2d(num_features=2*base),
                                   nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=2*base,out_channels=4*base,
                                             kernel_size=4,stride=2,padding=1,bias=False),
                                   nn.BatchNorm2d(num_features=4*base),
                                   nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=4*base,out_channels=8*base,
                                             kernel_size=4,stride=2,padding=1,bias=False),
                                   nn.BatchNorm2d(num_features=8*base),
                                   nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=8*base,out_channels=1,
                                             kernel_size=4,stride=1,padding=0,bias=False),
                                   #nn.Sigmoid())
                                    )
    def forward(self,x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        return x
    
if __name__=="__main__":
    # generator:
    net_g = Generator()
    print(net_g)
    z = torch.randn((100,100,1,1),requires_grad=False)
    with torch.no_grad():
        g_out = net_g(z)
    
    # discriminator
    net_d = Discriminator()
    print(net_d)
    d_out = net_d(g_out)
    
