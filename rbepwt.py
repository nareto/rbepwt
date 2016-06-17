#!/usr/bin/env python
import ipdb
import copy
import PIL
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pywt
import pickle
from skimage.segmentation import felzenszwalb 

_DEBUG = False

cdf97_an_lo = np.array([0.026748757411, -0.016864118443, -0.078223266529, 0.266864118443,\
                        0.602949018236,	0.266864118443,	-0.078223266529,-0.016864118443, \
                        0.026748757411])

cdf97_an_hi = np.array([0, 0.091271763114, -0.057543526229,-0.591271763114,1.11508705,\
                        -0.591271763114,-0.057543526229,0.091271763114,0 ])

cdf97_syn_lo = np.array([0,-0.091271763114,-0.057543526229,0.591271763114,1.11508705,\
                         0.591271763114	,-0.057543526229,-0.091271763114,0])

cdf97_syn_hi = np.array([0.026748757411,0.016864118443,-0.078223266529,-0.266864118443,\
                         0.602949018236,-0.266864118443,-0.078223266529,0.016864118443,\
                         0.026748757411])

cdf97 = pywt.Wavelet('cdf97', [cdf97_an_lo,cdf97_an_hi,cdf97_syn_lo,cdf97_syn_hi])


def rotate(vector,theta):
    """Rotates a 2D vector counterclockwise by theta"""
    
    matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return(np.dot(matrix,vector))

def neighborhood(coord,level):
    """Returns N_ij^level = {(k,l) s.t. max{|k-i|, |l-j| <= 2^(level-1)} } where (i,j) == coord"""
    
    ret = set()
    i,j = coord
    for row_offset in range(-2**(level-1),2**(level-1)+1):
        for col_offset  in range(-2**(level-1),2**(level-1)+1):
            k,l = i+row_offset,j+col_offset
            ret.add((k,l))
    return(ret)

def full_decode(wavelet_details_dict,wavelet_approx,label_img,wavelet):
    """Returns the decoded image, without using information obtained from encoding (i.e. all paths are recomputed)"""

    print("\n--FULL DECODE")
    levels = len(wavelet_details_dict)
    li_inst = Image()
    li_inst.read_array(label_img)
    rb_inst = Rbepwt(li_inst,levels,wavelet)
    segm_inst = Segmentation(li_inst)
    segm_inst.label_img = label_img
    segm_inst.compute_label_dict()
    rb_inst.img.segmentation = segm_inst
    rb_inst.img.has_segmentation = True
    rb_inst.encode(onlypaths=True)
    for lev, wdetail in wavelet_details_dict.items():
        rb_inst.wavelet_details[lev] = wdetail
    rb_inst.region_collection_dict[levels+1] = wavelet_approx
    decoded_region_collection = rb_inst.decode()
    decoded_img = np.zeros_like(label_img)
    for coord,value in decoded_region_collection.points.items():
        decoded_img[coord] = value
    return(decoded_img)

def ispowerof2(n):
    if n < 1:
        return(False)
    k = int(n)
    while k == int(k):
        k /= 2
    if k != 0.5:
        return(False)
    else:
        return(True)
    
class Image:
    def __init__(self):
        self.has_segmentation = False
        self.has_psnr = False
        self.has_decoded_img = False

    def __getitem__(self, idx):
        return(self.img[idx])

    def read(self,filepath):
        self.img = skimage.io.imread(filepath)
        self.size = self.img.size
        self.imgpath = filepath
        self.shape = self.img.shape
        self.pict = Picture()
        self.pict.load_array(self.img)

    def read_array(self,array):
        self.img = array
        self.size = self.img.size
        self.shape = self.img.shape
        self.pict = Picture()
        self.pict.load_array(self.img)

    def info(self):
        img = PIL.Image.open(self.imgpath)
        print(img.info, "\nshape: %s\ntot pixels: %d\nmode: %s" % (img.size,img.size[0]*img.size[1],img.mode))
        img.close()

    def segment(self,method='felzenszwalb',**args):
        self.segmentation = Segmentation(self.img)
        if method == 'felzenszwalb':
            if 'scale' in args.keys():
                scale = args['scale']
            else:
                scale = 200
            if 'sigma' in args.keys():
                sigma = args['sigma']
            else:
                sigma = 0.8
            if 'min_size' in args.keys():
                min_size = args['min_size']
            else:
                min_size = 10
            self.label_img, self.label_pict = self.segmentation.felzenszwalb(scale,sigma,min_size)
        self.has_segmentation = True
        
    def encode_rbepwt(self,levels, wavelet):
        if not ispowerof2(self.img.size):
            raise Exception("Image size must be a power of 2")
        self.rbepwt_levels = levels
        self.rbepwt = Rbepwt(self,levels,wavelet)
        self.rbepwt.encode()

    def decode_rbepwt(self):
        self.decoded_region_collection = self.rbepwt.decode()
        self.decoded_img = np.zeros_like(self.img)
        for coord,value in self.decoded_region_collection.points.items():
            self.decoded_img[coord] = value
        self.decoded_pict = Picture()
        self.decoded_pict.load_array(self.decoded_img)
        self.has_decoded_img = True
        self.has_psnr = False
        
    def psnr(self):
        """Returns PSNR (peak signal to noise ratio) of decoded image vs. original image"""
        
        if self.has_psnr:
            return(self.psnr)
        else:
            if not self.has_decoded_img:
                raise Exception("No decoded img to compute PSNR of")
            mse = np.sum((self.img - self.decoded_img)**2)
            mse /= self.img.size
            maxvalue = self.img.max()
            return(20*np.log2(maxvalue/mse))

    def nonzero_coefs(self):
        ncoefs = 0
        for level,arr in self.rbepwt.wavelet_details.items():
            ncoefs += arr.nonzero()[0].size
        ncoefs += self.rbepwt.region_collection_dict[self.rbepwt_levels+1].values.nonzero()[0].size
        return(ncoefs)
        
    def save(self,filepath):
        f = open(filepath,'wb')
        pickle.dump(self.__dict__,f,3)
        f.close()

    def load(self,filepath):
        f = open(filepath,'rb')
        tmpdict = pickle.load(f)
        f.close()
        self.__dict__.update(tmpdict)
        self.has_segmentation = True
        self.has_psnr = True
        self.has_decoded_img = True
        try:
            self.label_img
        except AttributeError:
            self.has_segmentation = False
        try:
            self.psnr_img
        except AttributeError:
            self.has_psnr = False
        try:
            self.decoded_img
        except AttributeError:
            self.has_decoded_img = False
            
        
    def show(self):
        self.pict.show('Original image')

    def show_decoded(self):
        self.decoded_pict.show('Decoded image')

    def save_decoded(self,filepath):
        self.decoded_pict.show('Decoded image', filepath=filepath)
        
    def show_segmentation(self):
        #self.label_pict.show(plt.cm.hsv)
        self.segmentation.show()

class Picture:
    def __init__(self):
        self.array = None
        self.mpl_fig = None

    def load_array(self,array):
        self.array = array

    def load_mpl_fig(self,mpl_fig):
        self.mpl_fig = mpl_fig

    def __save_or_show__(self,fig,filepath=None):
        if filepath is None:
            fig.show()
        else:
            fig.savefig(filepath)
        
    def show(self,title=None,colormap=plt.cm.gray,filepath=None):
        """Shows self.array or self.mpl"""
    
        if self.array != None:
            fig = plt.figure()
            plt.imshow(self.array, cmap=colormap, interpolation='none')
            plt.axis('off')
            if title != None:
                plt.title(title)
            self.__save_or_show__(fig,filepath)
        elif self.mpl_fig != None:
            self.__save_or_show__(self.mpl_fig,filepath)


class Segmentation:
    
    def __init__(self,image):
        self.img = image
        self.has_label_dict = False

    def felzenszwalb(self,scale,sigma,min_size):
        self.label_img = felzenszwalb(self.img, scale=float(scale), sigma=float(sigma), min_size=int(min_size))
        self.compute_label_dict()
        self.label_pict = Picture()
        self.label_pict.load_array(self.label_img)
        return(self.label_img,self.label_pict)

    def compute_label_dict(self):
        self.label_dict = {}
        for idx,label in np.ndenumerate(self.label_img):
            if label not in self.label_dict:
                self.label_dict[label] = Region([idx],[self.img[idx]])
            else:
                self.label_dict[label] += Region([idx],[self.img[idx]])
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

    def show(self):
        fig = plt.figure()
        plt.imshow(self.label_img,interpolation='none',cmap=plt.cm.hsv)
        self.pict = Picture()
        self.pict.load_mpl_fig(fig)
        self.pict.show()

class Region:
    """Region of points, which can always be thought of as a path since points are ordered"""
    
    def __init__(self, base_points, values=None, compute_dict = True):
        #if type(base_points) != type([]):
        #    raise Exception('The points to init the Path must be a list')
        if values is not None and len(base_points) != len(values):
            raise Exception('Input points and values must be of same length')
        self.points = {}
        self.base_points = tuple(base_points)
        self.trivial = False
        if values is None or len(values) == 0:
            self.values = np.array(len(base_points)*[None])
            self.no_values = True
            if len(base_points) == 0:
                self.trivial = True
        else:
            self.values = np.array(values)
            self.no_values = False
        if compute_dict:
            self.__init_dict_and_extreme_values__()
        
    def __init_dict_and_extreme_values__(self):
        self.start_point,self.top_left,self.bottom_right = None,None,None
        if not self.trivial:
            self.top_left = self.base_points[0]
            self.bottom_right = self.base_points[0]
            self.start_point = self.base_points[0]
            bp = self.base_points
            for i in range(len(bp)):
                if self.no_values:
                    self.points[bp[i]] = None
                else:
                    self.points[bp[i]] = self.values[i]
                row,col = bp[i]
                self.top_left = min(row,self.top_left[0]),min(col,self.top_left[1])
                self.bottom_right = max(row,self.bottom_right[0]),max(col,self.bottom_right[1])
                if row < self.start_point[0] or (row == self.start_point[0] and col < self.start_point[1]):
                    self.start_point = (row,col)

    def __getitem__(self, key):
        return((self.base_points[key],self.values[key]))

    def __add__(self,region):
        basepoints = self.base_points + region.base_points
        values = np.concatenate((self.values,region.values))
        newregion = Region(basepoints,values,False)
        newregion.points = {**self.points, **region.points} #syntax valid from python3.5. see http://stackoverflow.com/questions/38987
        sp1,sp2 = self.start_point, region.start_point
        tl1,br1,tl2,br2 = self.top_left, self.bottom_right, region.top_left, region.bottom_right
        if tl1 == None:
            newregion.top_left = tl2
            newregion.bottom_right = br2
        elif tl2 == None:
            newregion.top_left = tl1
            newregion.bottom_right = br1
        else:
            newregion.top_left = min(tl1[0],tl2[0]),min(tl1[1],tl2[1])
            newregion.bottom_right = max(br1[0],br2[0]),max(br1[1],br2[1])
        if sp1 == None:
            newregion.start_point = sp2
        elif sp2 == None:
            newregion.start_point = sp1
        elif sp1[0] < sp2[0] or (sp1[0] == sp2[0] and sp1[1] < sp2[1]):
            newregion.start_point = sp1
        else:
            newregion.start_point = sp2
        return(newregion)

    def __len__(self):
        return(len(self.points))

    def __iter__(self):
        self.__iter_idx__ = -1
        return(self)

    def __next__(self):
        self.__iter_idx__ += 1
        if self.__iter_idx__ >= len(self):
            raise StopIteration
        return((self.base_points[self.__iter_idx__],self.values[self.__iter_idx__]))

    def update_dict(self):
        self.__init_dict_and_extreme_values__()
    
    def add_point(self,coord,value):
        lbp = list(self.base_points)
        lbp.append(coord)
        self.base_points = tuple(lbp)
        lv = list(self.values)
        lv.append(value)
        self.values = tuple(lv)
        self.points[coord] = value
        i,j = coord
        self.top_left = min(i,self.top_left[0]),min(j,self.top_left[1])
        self.bottom_right_ = max(i,self.bottom_right[0]),max(j,self.bottom_right[1])
    
    def easy_path(self,level,inplace=False):
        start_point = self.start_point
        if len(self) == 1:
            self.permutation = [0]
            return(self)
        elif len(self) == 0:
            self.permutation = None
            return(self)
        bp = tuple(filter(lambda point: point != start_point,self.base_points))
        avaiable_points = set(bp)
        if len(bp) == 0:
            return(Region([start_point],[self.points[start_point]]))
        new_path_permutation = [self.base_points.index(start_point)]
        new_base_points = [start_point]
        new_values = np.array(self.points[start_point])
        cur_point = start_point
        found = 0
        prefered_direc = np.array((0,1))
        while cur_point != None:
            chosen_point = None
            min_dist = None
            candidate_points = set()
            k = 1
            while not candidate_points and avaiable_points:
                neigh = neighborhood(cur_point,k)
                neigh.remove(cur_point)
                candidate_points = avaiable_points.intersection(neigh)
                k+=1
            for candidate in candidate_points:
                dist = np.linalg.norm(np.array(cur_point) - np.array(candidate))
                if  min_dist == None or dist < min_dist:
                    min_dist = dist
                    chosen_point  = candidate
                elif min_dist == dist:
                    tmp_point = np.array(cur_point) + prefered_direc
                    v1 = np.array(chosen_point) - np.array(cur_point)
                    v2 = np.array(candidate) - np.array(cur_point)
                    sp1 = np.dot(v1,prefered_direc)
                    sp2 = np.dot(v2,prefered_direc)
                    if sp2 > sp1:
                        chosen_point = candidate
                    elif sp2 == sp1:
                        prefered_direc = rotate(prefered_direc,- np.pi/2)
                        sp1 = np.dot(v1,prefered_direc)#
                        sp2 = np.dot(v2,prefered_direc)
                        if sp2 > sp1:
                            chosen_point = candidate
            if chosen_point != None:
                found += 1
                #new_path.add_point(chosen_point,self.points[chosen_point])
                new_base_points.append(chosen_point)
                new_values = np.append(new_values,self.points[chosen_point])
                avaiable_points.remove(chosen_point)
                prefered_direc = np.array(chosen_point) - np.array(cur_point)
                cur_point = chosen_point
                new_path_permutation.append(self.base_points.index(chosen_point))
                if not avaiable_points:
                    break
            else:
                ipdb.set_trace()
                print("This shouldn't happen! ", cur_point)
        if inplace:
            self.base_points = tuple(new_base_points)
            self.values = new_values
            #TODO: is the following needed?
            self.permutation = new_path_permutation
        new_path = Region(new_base_points, new_values)
        new_path.permutation = new_path_permutation
        if _DEBUG:
            print("EASY PATH: permutation -- ", new_path.permutation)
        return(new_path)
            
    def reduce_points(self,skip_first=False):
        if len(self) == 0 or (len(self) == 1 and skip_first == True):
            return(Region([]))
        elif len(self) == 1 and skip_first == False:
            return(self)
        new_region = Region([])
        idx = 0
        for point,value in self:
            if idx % 2 == skip_first:
                new_region += Region([point],[value])
            idx += 1
        if len(new_region) == 0:
            new_region.no_values = True
        return(new_region)

    def show(self,point_size=5):
        fig = plt.figure()
        n = self.bottom_right[0] - self.top_left[0]
        m = self.bottom_right[1] - self.top_left[1]
        i,j = self.base_points[0]
        xp,yp = j, n-i
        random_color = tuple([np.random.random() for i in range(3)])
        plt.plot([xp],[yp], '+', ms=2*point_size,mew=10,color=random_color)
        for coord in self.base_points[1:]:
            i,j = coord
            x,y = j,n-i
            plt.plot([xp,x],[yp,y], '-x', linewidth=2, color=random_color)
            xp,yp = x,y
        self.pict = Picture()
        self.pict.load_mpl_fig(fig)
        self.pict.show()

        
class RegionCollection:
    """Collection of Regions"""
    
    def __init__(self,  *regions, copy_regions=True):
        self.subregions = {} #TODO: change this to list []. So there's no ambiguity in iterating over it
        self.nregions = 0
        self.region_lengths = []
        self.values = np.array([])
        self.base_points = []
        self.points = {}
        self.copy_regions = copy_regions
        points = []
        self.no_regions = False
        if len(regions) == 0:
            self.no_regions = True
            self.top_left,self.bottom_right = None, None
        else:
            self.top_left = regions[0].top_left
            self.bottom_right = regions[0].bottom_right
        for r in regions:
            for coord in r.base_points:
                if coord in points:
                    raise Exception("Conflicting coordinates in regions")
                else:
                    points.append(coord)
        for r in regions:
            self.add_region(r)
        
    def __len__(self):
        return(self.nregions)
            
    def __getitem__(self,key):
        if type(key) == type(()):
            return(self.points[key])
        elif type(key) == type(0):
            return(self.subregions[key])
                
    def __iter__(self):
        self.__iter_idx__ = -1
        return(self)

    def __next__(self):
        self.__iter_idx__ += 1
        if self.__iter_idx__ >= len(self):
            raise StopIteration
        return((self.__iter_idx__, self.subregions[self.__iter_idx__]))
            
    def add_region(self,region):
        for coord in region.base_points:
            if coord in self.base_points:
                raise Exception("Conflicting coordinates in regions")
        if self.copy_regions:
            newregion = copy.deepcopy(region)
        else:
            newregion = region
        if self.no_regions:
            if region.no_values:
                self.top_left,self.bottom_right = None,None
            else:
                self.top_left, self.bottom_right = region.top_left,region.bottom_right
        if not region.no_values:
            self.top_left = min(self.top_left[0],region.top_left[0]), min(self.top_left[1],region.top_left[1])
            self.bottom_right = max(self.bottom_right[0],region.bottom_right[0]), max(self.bottom_right[1],region.bottom_right[1])
        self.subregions[self.nregions] = newregion
        self.values = np.concatenate((self.values,newregion.values))
        self.base_points += newregion.base_points
        self.region_lengths.append(len(region))
        self.points = {**self.points, **region.points}
        self.nregions += 1
        self.no_regions = False

    def reduce(self,values):
        """Returns wavelet details for current level and a new region collection for the next"""
        
        new_region_collection = RegionCollection()
        skipped_prev,prev_had_odd_length = False, False
        prev_region_length = 0
        for key,subregion in self:
            if skipped_prev + prev_had_odd_length == True:
                skip_first = True
            else:
                skip_first = False
            skipped_prev = skip_first
            prev_had_odd_length = len(subregion) % 2
            newregion = subregion.reduce_points(skip_first)
            newregion.values = values[prev_region_length:prev_region_length + len(newregion)]
            newregion.update_dict()
            prev_region_length += len(newregion)
            newregion.generating_permutation = subregion.permutation
            #ipdb.set_trace()
            new_region_collection.add_region(newregion)
        #print("\n\n",len(new_region_collection.points))
        return(new_region_collection)

    def expand(self,values, upper_region_collection, wavelet):
        """Returns wavelet approximation coefficients for current level and new region collection for the previous"""

        new_region_collection = RegionCollection()
        prev_length = 0
        for key,subregion in self:
            #print("\n\n", key,subregion.points,"\n\n")
            upper_region = upper_region_collection[key]
            #if len(subregion) >= 1:
            if len(upper_region) >= 1:
                if subregion.generating_permutation == None:
                    ipdb.set_trace()
                subr_values = values[prev_length:prev_length+len(upper_region)]
                #print("\n\n", key,upper_region.points,subr_values,"\n\n")
                prev_length += len(upper_region)
                invperm = sorted(range(len(upper_region)), key = lambda k: subregion.generating_permutation[k])
                if _DEBUG:
                    print("EXPAND: invperm %s" % invperm)
                new_base_points = []
                new_subr_values = np.array([])
                for k in invperm:
                    new_base_points.append(upper_region.base_points[k])
                    new_subr_values = np.append(new_subr_values, subr_values[k])
                region_to_add = Region(new_base_points,new_subr_values)
            else:
                region_to_add = upper_region
                prev_length += len(upper_region)
            new_region_collection.add_region(region_to_add)
        return(new_region_collection)

    def show(self,title=None,point_size=5):
        fig = plt.figure()
        if title != None:
            plt.title(title)
        tl,br = self.top_left, self.bottom_right
        n,m = br[0] - tl[0], br[1] - tl[1]
        border = 0.5
        plt.xlim([tl[1] - border, br[1] + border])
        plt.ylim([tl[0] - border, br[0] + border])
        fig.gca().invert_yaxis()
        for key,subr in self:
            if not subr.trivial:
                random_color = tuple(np.random.random(3)*0.5)
                yp,xp = subr.base_points[0]
                plt.plot([xp],[yp], '+', ms=2*point_size,mew=10,color=random_color)
                for p in subr.base_points[1:]:
                    y,x = p
                    plt.plot([xp,x],[yp,y], '-x', linewidth=0.5, color=random_color)
                    xp,yp = x,y
        self.pict = Picture()
        self.pict.load_mpl_fig(fig)
        self.pict.show()

    def show_values(self,title=None):
        fig = plt.figure()
        if title != None:
            plt.title(title)
        plt.plot(self.values)
        self.pict = Picture()
        self.pict.load_mpl_fig(fig)
        self.pict.show()
        
class Rbepwt: 
    def __init__(self, img, levels, wavelet):
        if 2**levels > img.size:
            raise Exception('2^levels must be smaller or equal to the number of pixels in the image')
        if type(img).__name__ != type(Image()).__name__:
            raise Exception('First argument must be an Image instance')
        self.img = img
        self.levels = levels
        self.has_encoding = False
        self.wavelet = wavelet
        
    def encode(self,onlypaths=False):
        wavelet=self.wavelet
        if not self.img.has_segmentation:
            self.img.segment()
        regions = self.img.segmentation.label_dict.values()
        self.region_collection_dict = {1: RegionCollection(*tuple(regions))}
        if onlypaths:
            self.region_collection_dict[1].values = np.zeros(len(regions))
        self.wavelet_details = {}
        for level in range(1,self.levels+1):
            level_length = 0
            paths_at_level = []
            cur_region_collection = self.region_collection_dict[level]
            for key, subregion in cur_region_collection:
                level_length += len(subregion)
                paths_at_level.append(subregion.easy_path(level,inplace=True))
            cur_region_collection = RegionCollection(*paths_at_level)
            if onlypaths:
                wlen = len(cur_region_collection.values)/2
                wapprox,wdetail = np.zeros(wlen),np.zeros(wlen)
            else:
                wapprox,wdetail = pywt.dwt(cur_region_collection.values, wavelet)
            #print("wdetail size: %d" % wdetail.size)
            self.wavelet_details[level] = wdetail
            self.region_collection_dict[level+1] = cur_region_collection.reduce(wapprox)
            print("\n-- ENCODING: finished working on level %d" % level)
            if _DEBUG:
                for key, subr in self.region_collection_dict[level]:
                    print("ENCODING: subregion %s has base points %s" % (key,subr.base_points))
                    print("ENCODING: subregion %s has base values %s" % (key,subr.values))
                #print("ENCODING: self.region_collection_dict[level].values %s" % self.region_collection_dict[level].values)
                print("ENCODING: self.wavelet_details[level] %s" % self.wavelet_details[level])
                print("ENCODING: self.region_collection_dict[level+1].values %s" % self.region_collection_dict[level+1].values)
        self.has_encoding = True
            
    def decode(self):
        wavelet=self.wavelet
        if not self.has_encoding:
            raise Exception("There is no saved encoding to decode")
        nonzerocoefs = self.region_collection_dict[self.levels+1].values.nonzero()[0].size
        cur_region_collection = self.region_collection_dict[self.levels+1]
        values = self.region_collection_dict[self.levels+1].values
        for level in range(self.levels,0, -1):
            cur_region_collection = self.region_collection_dict[level+1]
            cur_region_collection.values = values #APPLY PERMUTATION FIRST
            wdetail,wapprox = self.wavelet_details[level], cur_region_collection.values
            nonzerocoefs += wdetail.nonzero()[0].size
            values = pywt.idwt(wapprox, wdetail, wavelet)
            upper_region_collection = self.region_collection_dict[level]
            new_region_collection = cur_region_collection.expand(values,upper_region_collection,wavelet)
            values = new_region_collection.values
            #cur_region_collection = new_region_collection
            print("\n--DECODING: finished working on level %d " %level)
            if _DEBUG:
                print("DECODING: cur_region_collection.base_points = %s" % cur_region_collection.base_points)
                print("DECODING: cur_region_collection.values = %s" % cur_region_collection.values)
                print("DECODING: self.wavelet_details[level] = %s" % self.wavelet_details[level])
                print("DECODING: new_region_collection.base_points = %s" % new_region_collection.base_points)
                print("DECODING: new_region_collection.values = %s" % new_region_collection.values)
        print("NONZEROCOEFS: %d" % nonzerocoefs)
        return(new_region_collection)

    def threshold_coefs(self,ncoefs):
        """Sets to 0 all but the ncoefs (among detail and approximation coefficients) with largest absolute value"""
        
        wav_detail = self.wavelet_details[1]
        lev_length = len(wav_detail)
        flat_coefs = np.stack((np.ones(lev_length),wav_detail))
        for lev in range(2,self.levels+1):
            wav_detail = self.wavelet_details[lev]
            lev_length = len(wav_detail)
            flat_coefs = np.append(flat_coefs,np.stack((lev*np.ones(lev_length),wav_detail)),1)
        wapprox_rc = self.region_collection_dict[self.levels + 1]
        wav_approx = wapprox_rc.values
        lev_length = len(wav_approx)
        flat_coefs = np.append(flat_coefs,np.stack(((self.levels+1)*np.ones(lev_length),wav_approx)),1)
        sorted_idx = np.argsort(abs(flat_coefs[1,:]))
        flat_thresholded_coefs = np.zeros_like(flat_coefs)
        flat_thresholded_coefs[0,:] = flat_coefs[0,:]
        counter = 0
        for idx in reversed(sorted_idx):
            flat_thresholded_coefs[1,idx] = flat_coefs[1,idx]
            counter += 1
            if counter == ncoefs:
                break
        prev_len = 0
        for lev in range(1,self.levels+1):
            lev_length = len(self.region_collection_dict[lev + 1].base_points)
            for k in range(lev_length):
                self.wavelet_details[lev][k] = flat_thresholded_coefs[1,prev_len + k]
            prev_len += lev_length
        for k in range(len(wapprox_rc.base_points)):
            self.region_collection_dict[self.levels +1].values[k] = flat_thresholded_coefs[1,prev_len +k]
            
    def threshold_coeffs_by_value(self,threshold,threshold_type='hard'): #TODO: never tested this
        for level in range(2,self.levels+2):
            if threshold_type == 'hard':
                idx = self.wavelet_details[level] > threshold
                self.wavelet_details[level] = self.wavelet_details[level][idx]

    def flat_wavelet(self):
        """Returns an array with all wavelet coefficients sequentially"""

        ret = np.array([])
        for lev in range(1,self.levels+1):
            wavelet_at_level = self.wavelet_details[lev]
            print(lev,wavelet_at_level.size)
            ret = np.append(ret,wavelet_at_level)
        ret = np.append(ret,self.region_collection_dict[self.levels+1].values)
        return(ret)
                
    def show_wavelets(self,levels=None,show_approx=True):
        if levels == None:
            levels = range(1,self.levels+1)
        for level in levels:
            self.show_wavelet_detail_at_level(level)
        if show_approx==True:
            self.region_collection_dict[self.levels+1].show_values('Wavelet approximation coefficients')
            
    def show_wavelet_detail_at_level(self,level):
        if level in self.wavelet_details:
            fig = plt.figure()
            plt.title('Wavelet detail coefficients at level %d ' % level)
            plt.plot(self.wavelet_details[level])
            #for key,subr in self.region_collection_dict[level]:
            #    print(key)
            #    miny,maxy = min(self.wavelet_details[level]),max(self.wavelet_details[level])
            #    x = self.region_collection_dict[level].region_lengths[key]
            #    #plt.plot([x,x],[miny,maxy],'r') #TODO: is this correct? should we use x/2 instead?
            self.pict = Picture()
            self.pict.load_mpl_fig(fig)
            self.pict.show()
    
    def show(self,levels=None,point_size=5):
        if levels == None:
            levels = range(1,self.levels+1)
        for lev in levels:
            if lev in self.region_collection_dict.keys():
                self.region_collection_dict[lev].show('Region collection at level %d' % lev)
        
            
