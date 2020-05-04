#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 10:40:39 2018

@author: yellow
"""

import net
import numpy as np
import torch
from torch.autograd import Variable
from astropy.io import fits
from PIL import Image
class Load_img(object):
    def __init__(self):
        pass

    def norm(self,img):
        if np.max(img) > np.min(img):
            img = (img - np.min(img))/(np.max(img) - np.min(img)) #normalization
        img -= np.mean(img)  # take the mean
        if np.std(img) > 0:
            img /= np.std(img)  #standardization
        img = np.array(img,dtype='float32')
        return img
                
    def read_fits(self,path):
        hdu = fits.open(path)
        img = hdu[0].data
        img = np.array(img,dtype = np.float32)
        hdu.close()
        return img

    def read_png(self,path):
        img = Image.open(path)
        img = img.resize((512,512),Image.BILINEAR)
        img = np.array(img,dtype = np.float32)
        return img

class Measure(Load_img):
    def __init__(self):
        self.vgg_model = net.Vgg16()
        if torch.cuda.is_available():
            print('=> Use CUDA')
    def pre_data(self,img):
        image = Load_img.norm(self,img)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        image = image.unsqueeze(0)
        image = torch.cat((image,image,image),1)
        return image

    def gram_matrix(self,y):
        """
        for each channel  in C of the feature map:
            element-wise multiply and sum together 
        return a C*C matrix
        """
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w*h)
        features_t = features.transpose(1,2)
        gram = features.bmm(features_t) /(ch*h*w)
        return gram
    
    def Preception(self,or_img,de_img):
        or_img3 = self.pre_data(or_img)
        de_img3 = self.pre_data(de_img)
        if torch.cuda.is_available():
            self.vgg_model.cuda()
            or_img3 = Variable(or_img3).cuda()
            de_img3 = Variable(de_img3).cuda()
            #print('for debug')
        feature_or = self.vgg_model(or_img3)
        feature_de = self.vgg_model(de_img3)
        if torch.cuda.is_available():
            or_img3 = torch.Tensor.cpu(or_img3)
            de_img3 = torch.Tensor.cpu(de_img3)
        L = 0
        x1 = self.gram_matrix(feature_or[3].detach())
        x1 = x1.reshape(1,-1)[0]
        x2 = self.gram_matrix(feature_de[3].detach())
        x2 = x2.reshape(1,-1)[0]
        L = torch.dot(x1,x2)/(torch.norm(x1)*torch.norm(x2))
        L = np.float32(L)
        return L            
