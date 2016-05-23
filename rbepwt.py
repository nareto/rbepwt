#!/usr/bin/env python

import PIL
import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb 

class Image:
    
    def __init__(self):
        pass
        #self.img,self.segmentation = [None]*2

    def read(self,filepath):
        self.img = skimage.io.imread(filepath)
        self.imgpath = filepath
        self.pict = Picture()
        self.pict.load_array(self.img)

    def info(self):
        img = PIL.Image.open(self.imgpath)
        print(img.info, "\nshape: %s\ntot pixels: %d\nmode: %s" % (img.size,img.size[0]*img.size[1],img.mode))
        img.close()

    def segment(self,method='felzenszwalb'):
        self.segmentation = Segmentation(self.img)
        if method == 'felzenszwalb':
            self.label_img, self.label_pict = self.segmentation.felzenszwalb()


    def rbepwt(self):
        pass

    def irbepwt(self):
        #return(reconstructed_image)
        pass

    def threshold_coeffs(self,threshold_type):
        pass

    def psnr(self):
        pass

    def show(self):
        self.pict.show()

    def show_segmentation(self):
        self.label_pict.show(plt.cm.hsv)

    
class Segmentation:

    def __init__(self,image):
        self.img = image

    def felzenszwalb(self,scale=200,sigma=0.8,min_size=10):
        self.label_img = felzenszwalb(self.img, scale=float(scale), sigma=float(sigma), min_size=int(min_size))
        self.label_pict = Picture()
        self.label_pict.load_array(self.label_img)
        return(self.label_img,self.label_pict)
    
    def estimate_perimeter(self):
        pass

class Path:

    def __init__(self, base_points, values):
        pass

    def find_path(self,method):
        pass

    def wavelet_transform(self,wavelet):
        pass

    def show(self):
        pass

class Picture:
    
    def __init__(self):
        self.array = None
        self.mpl_obj = None

    def load_array(self,array):
        self.array = array

    def load_mpl_obj(self,mpl_obj):
        self.mpl_obj = mpl_obj
        
    def show(self,colormap=plt.cm.gray):
        """Shows self.array or self.mpl"""
    
        if self.array != None:
            plt.imshow(self.array, cmap=colormap, interpolation='none')
            plt.axis('off')
            plt.show()
            
