#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:39:25 2019

@author: heloise
"""

from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
#from feature import *
from helper import *
import random as rd
from scipy import misc

root_dir = "training/"
image_dir = root_dir + "images/"
files = os.listdir(image_dir)
files.sort()
#print(files)
n = min(100, len(files))



for i in range (n):
    img = load_image(image_dir + files[i])
#    fig = plt.figure()
#    plt.gray()  # show the filtered result in grayscale
#    ax1 = fig.add_subplot(121)  # left side
#    ax2 = fig.add_subplot(122)  # right side
    result = gaussian_filter(img, sigma=0.9)
#    ax1.imshow(img)
#    ax2.imshow(result)
#    plt.show()
    m=np.amax(result)
    mM=np.amin(result)
    result=(result-mM)/(m-mM)
    result = (result * 255).round().astype(np.uint8)
#    filename='gaussianfiltering/training/images/satImage_'+  '%.3d' % (i+1) +  '.png'
#    result.save(filename) 
    Image.fromarray(result).save('gaussianfiltering/training/images/satImage_'+  '%.3d' % (i+1) +  '.png')    






for image_idx in range (0,50):
    imageid = "test_" + str(image_idx+1)
    image_filename = 'test_set_images/' + imageid +'/'+imageid + ".png"
    print(image_filename)
    img = load_image(image_filename)
#    fig = plt.figure()
#    plt.gray()  # show the filtered result in grayscale
#    ax1 = fig.add_subplot(121)  # left side
#    ax2 = fig.add_subplot(122)  # right side
    result = gaussian_filter(img, sigma=0.9)
#    ax1.imshow(img)
#    ax2.imshow(result)
#    plt.show()
    m=np.amax(result)
    mM=np.amin(result)
    result=(result-mM)/(m-mM)
    result = (result * 255).round().astype(np.uint8)
#    filename='gaussianfiltering/training/images/satImage_'+  '%.3d' % (i+1) +  '.png'
#    result.save(filename) 
    Image.fromarray(result).save('gaussianfiltering/testing/'+ imageid +  '.png')   
