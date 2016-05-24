#!/usr/bin/env python

import PIL
import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb 

def concat_two_paths(instpath1,instpath2):
    points = []
    values = []
    for idx,val in instpath1.points.items():
        p,v = val
        points.append(p)
        values.append(v)
    for idx,val in instpath2.points.items():
        p,v = val
        points.append(p)
        values.append(v)
    newinst = Path(points,values)
    return(newinst)

def concat_paths(*paths):
    prev = Path([])
    for cur in paths:
        prev = concat_two_paths(prev,cur)
    return(prev)

class Image:
    def __init__(self):
        self.has_segmentation = False

    def __getitem__(self, idx):
        return(self.img[idx])

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
        self.has_segmentation = True
        
    def compute_rbepwt(self,levels=2):
        self.rbepwt = Rbepwt(self,levels)
        self.rbepwt.compute()

    def compute_irbepwt(self):
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

class Rbepwt:
    def __init__(self, img, levels=2):
        if type(img) != type(Image()):
            raise Exception('First argument must be an Image instance')
        self.img = img
        self.levels = levels

    def __init_path_data_structure__(self):
        if not self.img.has_segmentation:
            self.img.segment()
        label_dict = self.img.segmentation.compute_label_dict()
        self.paths = {}
        for label,points in label_dict.items():
            values = []
            for idx in points:
                i,j = idx
                values.append(self.img[i,j])
            self.paths[label] = Region(points,values)
        
    def find_path(self,method):
        self.path = self.points

    def compute(self):
        self.__init_path_data_structure__()

    def threshold_coeffs(self,threshold):
        pass
    
class Segmentation:
    def __init__(self,image):
        self.img = image
        self.has_label_dict = False

    def felzenszwalb(self,scale=200,sigma=0.8,min_size=10):
        self.label_img = felzenszwalb(self.img, scale=float(scale), sigma=float(sigma), min_size=int(min_size))
        self.label_pict = Picture()
        self.label_pict.load_array(self.label_img)
        return(self.label_img,self.label_pict)

    def compute_label_dict(self):
        #n,m = self.label_img.shape
        self.label_dict = {}
        for idx,label in np.ndenumerate(self.label_img):
            if label not in self.label_dict:
                self.label_dict[label] = [idx]
                #self.label_dict[label] = 
            else:
                self.label_dict[label].append(idx)
        self.has_label_dict = True
        return(self.label_dict)
                
    def compute_nlabels(self):
        try:
            self.nlabels = len(self.label_dict.keys())
        except AttributeError:
            labels = []
            for label in np.nditer(self.label_img):
                if label not in labels:
                    labels.append(label)
            self.nlabels = len(labels)
        return(self.nlabels)
            
    def estimate_perimeter(self):
        pass

class Region:
    """Region of points, which can always be thought of as a path since points are ordered"""
    
    def __init__(self, base_points, values=None):

        #if type(base_points) != type([]):
        #    raise Exception('The points to init the Path must be a list')
        if values != None and len(base_points) != len(values):
            raise Exception('Input points and values must be of same length')
        self.points = {}
        for i in range(len(base_points)):
            if values == None:
                self.points[i] = [base_points[i],None]
            else:
                self.points[i] = [base_points[i],values[i]]
                
    def __getitem__(self, key):
        return(self.points[key])

    def __len__(self):
        return(len(self.points))

    def __iter__(self):
        self.__iter_idx__ = -1
        return(self)

    def __next__(self):
        self.__iter_idx__ += 1
        if self.__iter_idx__ >= len(self):
            raise StopIteration
        return(self.points[self.__iter_idx__])
    
    def reduce_points(self):
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
            
