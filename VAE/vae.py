#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 09:49:58 2018

@author: huijian
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave, imresize

def kl_loss(mean, logvar):
    """
    mean and log_var are both Variables or at least Tensors
    """
    kl_loss = 0.5*torch.sum(torch.exp(logvar)+mean**2-1.-logvar,1)
    kl_loss = torch.sum(kl_loss)

    return kl_loss

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder,self).__init__()
        self.shared = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                #nn.Dropout(0.1),
                nn.Linear(hidden_dim,hidden_dim),
                nn.ReLU(),#nn.Tanh(),
                #nn.Dropout(0.1),
                )
            
        self.encode_mean = nn.Sequential(
                nn.Linear(hidden_dim, latent_dim),
                )
        self.encode_logvar = nn.Sequential(
                nn.Linear(hidden_dim,latent_dim))
    def forward(self,x):
        x = self.shared(x)
        mean = self.encode_mean(x)
        logvar = self.encode_logvar(x)
        return mean, logvar
        
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),#nn.Tanh(),
                #nn.Dropout(0.1),
                nn.Linear(hidden_dim,hidden_dim),
                nn.ReLU(),
                #nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid())
                
    def forward(self,x):
        x = self.main(x)
        return x
                
class VAE(nn.Module):
    def __init__(self, encoder,decoder):
        super(VAE,self).__init__()
        self.encode = encoder
        self.decode = decoder

    def _sample(self,mean,logvar):
        epsilon = Variable(torch.randn(mean.size(0), mean.size(1)))
        return mean+epsilon*torch.exp(logvar/2.)

    def forward(self,x):
        mean, logvar = self.encode(x)
        self.z_mean = mean
        self.z_logvar = logvar
        z = self._sample(mean, logvar)
        recon_x = self.decode(z)
        return recon_x, self.z_mean, self.z_logvar

def train_vae(vae, train, test, batch_size=16, epoch=100):
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(vae.parameters(),lr=1e-3, betas=(0.9,0.99))

    
    bce = nn.BCELoss(size_average=False)
    mse = nn.MSELoss(size_average=False)
    criterion = mse
    for i in range(epoch):
        vae.train()
        for idx, sample in enumerate(train_loader):
            image, label = sample
            image = Variable(image, requires_grad=False)
            image = image.view(image.size(0),-1)
            
            optimizer.zero_grad()
            recon_image,z_mean,z_logvar = vae(image)
            difference = criterion(recon_image,image)
            kl = kl_loss(mean=z_mean, logvar=z_logvar)
            
            loss = difference+kl
            loss.backward()
            optimizer.step()
            
        if (i+1)%1==0:
            vae.eval()
            image=image.view(image.size(0),1,28,28)
            recon_image = recon_image.view(recon_image.size(0),1,28,28)
            save_image(image.data, filename="./result/Epoch-{}-idx-{}_original_image.png".format(i+1,idx+1), nrow=4)
            save_image(recon_image.data, filename="./result/Epoch-{}-idx-{}_recon_image_average.png".format(i+1,idx+1), nrow=4)
                
        vae.eval()
        if (i+1)%1 == 0:
            loss = []
            for idx, samlpe in enumerate(test_loader):
                image, label = samlpe
                image = Variable(image,requires_grad=False)
                image = image.view(image.size(0),-1)
                
                recon_image,z_mean,z_logvar = vae(image)
                tmp_loss = kl_loss(mean=z_mean, logvar = z_logvar) + criterion(recon_image,image)
                
                loss.append(tmp_loss.data.numpy()[0])
            loss = np.sum(loss)/len(loss)
            print("Epoch:{}/{}, loss:{}".format(i+1,epoch,loss))

def plot_2d_manifold(vae,z_range=2,row=20,col=20):
    z = np.rollaxis(
            np.mgrid[z_range:-z_range:row*1j, z_range:-z_range:col*1j],
            0,3
            )
    z = z.reshape([-1,2]).astype(np.float32)
    z = Variable(torch.FloatTensor(z),requires_grad=False)
    recon_image = vae.decode(z)
    
    # to save the result
    img_h = 28
    img_w = 28
    resize_factor = 1.0
    
    new_h = int(img_h*resize_factor)
    new_w = int(img_w*resize_factor)
    
    final_image = np.zeros((new_h*row, new_w*col),dtype= np.float32)
    
    recon_image = recon_image.data.numpy()
    recon_image = recon_image.reshape(row*col, img_h, img_w)
    
    for idx, image in enumerate(recon_image):
        i = int(idx%col)
        j = int(idx/col)
        
        new_image = imresize(image, size=(new_h,new_w),interp="bicubic")
        final_image[j*new_h:j*new_h+new_h, i*new_w: i*new_w+new_w] = new_image
    
    imsave("./latent-space/manifold.png",final_image)

def discrete_cmap(N,base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    Note: 
        if base_map is a string or None, you can simply do
        return plt.cm.get_cmap(base_map,N)
        The following works for string, None, or a colormap instance:
    """
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0,1,N))
    cmap_name = base.name+str(N)
    return base.from_list(cmap_name, color_list, N)
    

def plot_scattered_image(z, idx, name="scattered_image.jpg"):
    plt.figure(figsize=(8,6))
    N = 10
    plt.scatter(z[:,0],z[:,1],c=idx,marker='o',edgecolor="none",cmap=discrete_cmap(N,'jet'))
    plt.colorbar(ticks=range(N))
    plt.grid(True)
    plt.savefig("./latent-space/"+name)

if __name__ == "__main__":

    batch_size = 16
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.MNIST("./data", train=True,download=True, transform = transform)
    test_dataset = torchvision.datasets.MNIST("./data",train=False,download=True, transform = transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    encoder = Encoder(input_dim = 28*28, hidden_dim = 256, latent_dim=2)
    decoder = Decoder(latent_dim = 2, hidden_dim = 256, output_dim=28*28)
    
    vae = VAE(encoder=encoder, decoder=decoder)
    
    if False:
        print("Load the model")
        vae = torch.load("./model/vae.pkl")
    if True:
        print("Training the model")
        train_vae(vae=vae, train=train_dataset, test=test_dataset, batch_size=batch_size,epoch=100)
    if True:
        print("Save the model")
        torch.save(vae,"./model/vae.pkl")
    if True:
        plot_2d_manifold(vae,z_range=2,row=20,col=20)
    if True:
        plt_data = DataLoader(train_dataset, batch_size=50000,shuffle=True)
        image, label = plt_data.__iter__().next()
        image = Variable(image)
        label = Variable(label)
        image = image.view(image.size(0),-1)
        z_mean, z_logvar = vae.encode(image)
        epsilon = Variable(torch.randn(z_mean.size(0), z_mean.size(1)))
        z= z_mean+epsilon*torch.exp(z_logvar/2.)
        plot_scattered_image(z.data.numpy(),label.data.numpy())
        