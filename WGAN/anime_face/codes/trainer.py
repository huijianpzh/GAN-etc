#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 21:51:36 2018

@author: huijian
"""

import torch
import torch.optim as optim
import torch.utils as utils
import torch.autograd as autograd
import torchvision

from skimage import io
import tqdm 
import numpy as np

def weight_init(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv")!=-1:
        m.weight.data.normal_(0.0,0.02)
    elif class_name.find("BatchNorm")!=-1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)

class train_wgan_gp(object):
    def __init__(self, disc, gen,z_dim = 100, file_path = "./model", cuda=False):
        
        self.z_dim = z_dim
        self.file_path = file_path
        
        # network
        self.disc = disc
        self.gen = gen
         
        if (self.disc != None) and (self.gen != None):
            self.disc.apply(weight_init)
        
        # set the device
        self.cuda = cuda
        self.device = torch.device("cuda" if cuda else "cpu" )
    
    def cal_gradient_penalty(self, real_data, fake_data, l = 10):
        
        batch_size = real_data.size(0)
        channels = real_data.size(1)
        height = real_data.size(2)
        width = real_data.size(3)
        
        # torch.randn: uniform distribution on the interval 0~1
        alpha = torch.rand((batch_size,1),device=self.device) # no need to move it to gpu
        alpha = alpha.expand(batch_size,real_data.nelement()/batch_size).contiguous().view(batch_size,channels,height,width)
        
        interpolates = alpha * real_data.detach() + ((1-alpha) * fake_data.detach())

    	    #requires_grad
        """
        This step is required for the autograd.grad to compute the gp
        """
        interpolates.requires_grad_(True)
        
        # get the gradients
        disc_interpolates = self.disc(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates,inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device) if self.cuda else torch.ones(disc_interpolates.size()),
                                  create_graph=True,retain_graph=True,only_inputs=True)[0]
        
        gradients = gradients.view(gradients.size(0),-1)
        gradients_penalty = l*((gradients.norm(2,dim=1)-1)**2).mean()
    
        return gradients_penalty
    
    def generate_image(self,file_name="test.png",gen_num=64,gen_search_num=512, z_mean=0, z_std=1.0,saved=True):
        # change the state
        self.gen.eval()
        self.disc.eval()
        
        z = torch.randn((gen_search_num,100,1,1),device = self.device).normal_(z_mean,z_std)
        with torch.no_grad():
            generated = self.gen(z)
            scores = self.disc(generated)
            
        _,idx = torch.topk(scores,k=gen_num,dim=0)
        
        if self.cuda: 
            generated = generated.cpu()
            idx = idx.cpu()
        
        generated = generated.numpy()
        result = []
        for i in idx:
            tmp = generated[i,:,:,:]
            tmp = torch.tensor(tmp, dtype=torch.float32)
            result.append(tmp)
        
        torchvision.utils.save_image(torch.stack(result),filename=file_name,normalize=True,range=(-1,1))
        return generated
        
    def train_model(self,dataset,epoch=300,batch_size=3, d_every=1, g_every=5):
        
        # define the dataset
        data_loader = utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
         
        # change the state
        if self.cuda:
            self.gen = self.gen.to(self.device)
            self.disc = self.disc.to(self.device)
            
        self.gen.train()
        self.disc.train()
        
        # define the optimizer
        optim_D = optim.Adam(self.disc.parameters(),lr=2e-4,betas=(0.5,0.99)) 
        optim_G = optim.Adam(self.gen.parameters(),lr=2e-4,betas=(0.5,0.99))
        
        # define one and mone
        one = torch.tensor([1],dtype=torch.float32)
        mone = one * -1
        
        if self.cuda:
            one = one.to(self.device)
            mone = mone.to(self.device)
        
        # iteration
        for e in range(epoch):
            for i, images in tqdm.tqdm(enumerate(data_loader)):
                
                # update Discriminator
                if (i+1) % d_every == 0:
                    for p in self.disc.parameters():
                        p.requires_grad=True
                    
                    optim_D.zero_grad()
                    # train with real
                    real_data = images["image"]
                    real_data = real_data.to(dtype=torch.float32)

                    if self.cuda:
                        real_data = real_data.to(self.device)
                        
                    D_real = self.disc(real_data)
                    D_real = D_real.mean()
                    D_real.backward(mone)
                    
                    # train with fake
                    z = torch.randn((batch_size,self.z_dim,1,1),dtype=torch.float32)
                    if self.cuda:
                        z = z.to(self.device)
                    
                    # fronze the gen
                    with torch.no_grad():
                        fake_data = self.gen(z)
                    
                    D_fake = self.disc(fake_data)
                    D_fake = D_fake.mean()
                    D_fake.backward(one)
                    
                    # train with gradient penalty 
                    gp = self.cal_gradient_penalty(real_data=real_data,fake_data=fake_data)
                    gp.backward()
                    
                    # compute
                    D_loss = (D_real - D_fake - gp) # E[D(x)] - E[D(G(z))] - gp
                    Wasserstein_D = D_real - D_fake
                    optim_D.step()
                
                
                # update Generator
                if (i+1) % g_every == 0:
                    for p in self.disc.parameters():
                        p.requires_grad = False
                    optim_G.zero_grad()
                        
                    z = torch.randn((batch_size,self.z_dim,1,1),dtype=torch.float32)
                    if self.cuda:
                        z = z.to(self.device)
                    fake_data = self.gen(z)
                    G = self.disc(fake_data)
                    G = G.mean()  # E[D(G(z))]
                    G.backward(mone)
                    
                    # compute 
                    G_loss = -G
                    optim_G.step()
                    
                if (i+1)%10==0:
                    if self.cuda:
                        D_loss = D_loss.cpu()
                        Wasserstein_D = Wasserstein_D.cpu()
                        G_loss = G_loss.cpu()
                    d = D_loss.detach().numpy()
                    g = G_loss.detach().numpy()
                    w = Wasserstein_D.detach().numpy()

                    print("-"*10)
                    print("epoch:{e:>5},iter:{i:>5}".format(e=e+1,i=i+1))
                    print("D_loss:{},G_loss:{},Wasserstein_D:{}".format(d,g,w))
                    print("-"*10)
                # generate images
                if (i+1)%10==0:                    
                    image = "../result"+ "/" + "epoch_" + str(e+1) + "_iters_" + str(i+1) + ".png"
                    self.generate_image(file_name=image,saved=True)
                    
            
            if (e+1)%1== 0:
                name = "dcgan_"
                self.save_model(name)
                if False:
                    name = "dcgan_" + "epoch_" + str(e+1) + "_"
                    self.save_model(name)
          
    def save_model(self,name = "dcgan_"):
        
        gen_name = self.file_path + "/" + name + "gen.pkl"
        disc_name = self.file_path + "/" + name + "disc.pkl"
        
        if self.cuda:
            self.gen = self.gen.cpu()
            self.disc = self.disc.cpu()
        
        torch.save(self.gen, gen_name)
        torch.save(self.disc, disc_name)
        
        if self.cuda:
            self.gen = self.gen.to(self.device)
            self.disc = self.disc.to(self.device)
            
        print("model saved!")
        return
        
    def restore_model(self,name="dcgan_"):
        
        gen_name = self.file_path + "/" + name + "gen.pkl"
        disc_name = self.file_path + "/" + name + "disc.pkl"
        
        self.gen = torch.load(gen_name)
        self.disc = torch.load(disc_name)
        
        for p in self.disc.parameters():
            p.requires_grad = True
        for p in self.gen.parameters():
            p.requires_grad = True
           
        """
        the model will be transfered to cpu in the self.train_model fn.
        """
        print("model restored!")
        return
    
