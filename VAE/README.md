# An implemantation of VAE

## Introduction
We use pytorch to write a VAE.
The MNIST dataset has been tested.

Here, we get the reconstructed pictures.
Meanwhile, the latent space obtained with encoder has been explored.
The below are the results for them:

### Reconstructed Pictures(original and reconstructed)
![](https://github.com/huijianpzh/GAN-etc/blob/master/VAE/result/Epoch-100-idx-3750_original_image.png)   ![](https://github.com/huijianpzh/GAN-etc/blob/master/VAE/result/Epoch-100-idx-3750_recon_image_average.png)
  
 
### Generated Picture
The dim of the latent code is set to be 2, and we give the value for z, [-2, 2]x[-2, 2].(using numpy.mgrid to do this)
![](https://github.com/huijianpzh/GAN-etc/blob/master/VAE/latent-space/manifold-epoch-100-network2.png)
### The latent codes of MNIST dataset
The latent codes are obtained with the MNIST dataset as input.
The z is computed as 
``` z_mean + torch.exp(z_logvar/2.)*epsilon ```.
 
 
 
![](https://github.com/huijianpzh/GAN-etc/blob/master/VAE/latent-space/scattered_image-epoch-100-network2.jpg)



During coding, we has read some codes from
* [Fast Forward Labs](https://github.com/fastforwardlabs/vae-tf)
* [hwalsuklee](https://github.com/hwalsuklee/tensorflow-mnist-VAE)

Note:
Different results will be produced when different network architectures are adopted.
Meanwhile, you should take care of the loss function. The parameter "size_average" should be set False.
(Thanks for the help from Dr Mengkun Du.)


## Referrences:
* [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)
* [Tutorials on Variational Autoencoder](https://arxiv.org/abs/1606.05908)
