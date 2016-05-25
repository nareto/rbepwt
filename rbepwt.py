#!/usr/bin/env python

import PIL
import skimage
import numpy as np
import matplotlib.pyplot as plt
import scipy
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
        self.shape = self.img.shape
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
        paths_at_first_level = {}
        for label,points in label_dict.items():
            values = []
            for idx in points:
                i,j = idx
                values.append(self.img[i,j])
            paths_at_first_level[label] = Region(points,values)
        self.paths[1] = paths_at_first_level
            
    def compute(self):
        self.__init_path_data_structure__()

    def threshold_coeffs(self,threshold):
        pass

    def show(self,level=1,point_size=5):
        fig = plt.figure()
        n,m = self.img.shape
        paths = self.paths[level]
        for label,path in paths.items():
            random_color = tuple([np.random.random() for i in range(3)])
            offset=0.3
            i,j = path.base_points[0]
            xp,yp = j,n-i
            plt.plot([xp],[yp], '+', ms=3*point_size,mew=10,color=random_color)
            for p in path.base_points[1:]:
                #y,x = p
                i,j = p
                x,y = j, n-i
                #plt.plot([x],[y], '.', ms=point_size,color=random_color)
                if max(abs(x-xp), abs(y-yp)) > 1:
                    #find out which is the indipendent variable
                    if x != xp:
                        ind, indp, dip, dipp = x,xp,y,yp
                    else:
                        ind, indp, dip, dipp = y,yp,x,xp                        
                    minind = min(ind,indp)
                    maxind = max(ind,indp)
                    if ind == minind:
                        mindip,maxdip = dip, dipp
                    else:
                        mindip,maxdip = dipp,dip
                    step_ind = (maxind - minind)/3
                    step_dip = (maxdip - mindip)/3
                    orthogonalvec_norm = np.sqrt((maxdip - mindip)**2 + (maxind - minind)**2)
                    orthogonalvec_ind = (maxdip - mindip)/orthogonalvec_norm
                    orthogonalvec_dip = (minind - maxind)/orthogonalvec_norm
                    indvec = [minind, minind+step_ind+offset*orthogonalvec_ind,\
                              minind+2*step_ind+offset*orthogonalvec_ind,maxind]
                    dipvec = [mindip, mindip+step_dip+offset*orthogonalvec_dip,\
                              mindip+2*step_dip+offset*orthogonalvec_dip,maxdip]
                    #plt.plot(indvec,dipvec,'x',color=random_color)
                    curve = scipy.interpolate.UnivariateSpline(indvec,dipvec,k=2)
                    splinerangestep = (maxind - minind)/10
                    splinerange = np.arange(minind,maxind+splinerangestep/2,splinerangestep)
                    if ind == x:
                        plt.plot(splinerange,curve(splinerange),'--',color="black")
                    else:
                        plt.plot(curve(splinerange),splinerange,'--',color="black")
                    #print("splinerange = %s \n xvec = %10s \tyvec = %10s\n x: %2d \t y: %2d \nxp: %2d \typ: %2d\n\n"\
                    #      % (splinerange,xvec,yvec,x,y,xp,yp))
                else:
                    plt.plot([xp,x],[yp,y], '-x', linewidth=0.5, color=random_color)
                xp,yp = x,y
        self.pict = Picture()
        self.pict.load_mpl_fig(fig)
        self.pict.show()

    
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
        top_left,bottom_right = base_points[0],base_points[0]
        for i in range(len(base_points)):
            if values == None:
                self.points[i] = [base_points[i],None]
            else:
                self.points[i] = [base_points[i],values[i]]
            row,col = base_points[i]
            if row <= top_left[0] and col <= top_left[1]:
                top_left = base_points[i]
            if row >= bottom_right[0] and col >= bottom_right[1]:
                bottom_right = base_points[i]
        self.top_left = top_left
        self.bottom_right= bottom_right
                
        self.base_points = base_points
        self.values = values
                
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
    
    def find_path(self,method):
        #self.path = self.points
        new_path = Path(self.base_points,self.values)

    def reduce_points(self):
        pass
    
    def wavelet_transform(self,wavelet):
        pass

    def show(self,point_size=5):
        fig = plt.figure()
        n = self.bottom_right[0] - self.top_left[0]
        m = self.bottom_right[1] - self.top_left[1]
        i,j = self.base_points[0]
        xp,yp = j, n-i
        random_color = tuple([np.random.random() for i in range(3)])
        plt.plot([xp],[yp], '+', ms=3*point_size,mew=10,color=random_color)
        for coord in self.base_points[1:]:
            i,j = coord
            x,y = j,n-i
            plt.plot([xp,x],[yp,y], '-x', linewidth=0.5, color=random_color)
            xp,yp = x,y
        self.pict = Picture()
        self.pict.load_mpl_fig(fig)
        self.pict.show()
        

class Picture:
    def __init__(self):
        self.array = None
        self.mpl_fig = None

    def load_array(self,array):
        self.array = array

    def load_mpl_fig(self,mpl_fig):
        self.mpl_fig = mpl_fig
        
    def show(self,colormap=plt.cm.gray):
        """Shows self.array or self.mpl"""
    
        if self.array != None:
            fig = plt.figure()
            plt.imshow(self.array, cmap=colormap, interpolation='none')
            plt.axis('off')
            fig.show()
        elif self.mpl_fig != None:
            self.mpl_fig.show()
            
