#!/usr/bin/env python
import ipdb
import copy
import PIL
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy
import pywt
import pickle
import queue
import skimage.filters
from skimage.segmentation import felzenszwalb 
from skimage.measure import compare_ssim
import sklearn.cluster as skluster

_DEBUG = False

def myround(x):
    return(np.floor(x+0.5))

def compare_wavelet_dicts(wd1,wd2):
    """Compares wavelet dictionaries in the format given by the rbepwt.wavelet_coefs_dict() method"""

    for level, coefs1 in wd1.items():
        coefs2 = wd2[level]
        for i in range(len(coefs1)):
            v1 = wd1[level][i]
            v2 = wd2[level][i]
            if v1 != v2:
                print("level: %2d\tindex: %4d\tvalue1: %5f\tvalue2: %5f" %(level,i,v1,v2))
                
def rotate(vector,theta):
    """Rotates a 2D vector counterclockwise by theta"""
    
    matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return(np.dot(matrix,vector))

def neighborhood(coord,level,mode='square',hole=False):
    """Returns N_ij^level = {(k,l) s.t. max{|k-i|, |l-j| <= 2^(level-1)} } where (i,j) == coord"""
    
    ret = []
    i,j = coord
    if mode == 'square':
        row_range = range(-2**(level-1),2**(level-1)+1)
        col_range = range(-2**(level-1),2**(level-1)+1)
        ret = [(i+roff,j+coff) for roff in row_range for coff in col_range]
    elif mode == 'cross':
        for row_offset in range(-2**(level-1),2**(level-1)+1):
            k = i+row_offset
            ret.append((k,j))
        for col_offset  in range(-2**(level-1),2**(level-1)+1):
            l = j+col_offset
            ret.append((i,l))
    else:
        raise Exception("Mode must be either square or cross")
    if hole:
        ret.remove(coord)
    return(ret)

def full_decode(wavelet_details_dict,wavelet_approx,label_img,wavelet,path_type='easypath'):
    """Returns the decoded image, without using information obtained from encoding (i.e. all paths are recomputed)"""

    print("\n--FULL DECODE--")
    levels = len(wavelet_details_dict)
    li_inst = Image()
    li_inst.read_array(label_img)
    rb_inst = Rbepwt(li_inst,levels,wavelet,path_type=path_type)
    segm_inst = Segmentation(li_inst)
    segm_inst.label_img = label_img
    segm_inst.compute_label_dict()
    rb_inst.img.segmentation = segm_inst
    rb_inst.img.has_segmentation = True
    rb_inst.encode(onlypaths=True)
    for lev, wdetail in wavelet_details_dict.items():
        rb_inst.wavelet_details[lev] = wdetail
    rb_inst.region_collection_at_level[levels+1] = wavelet_approx
    decoded_region_collection = rb_inst.decode()
    decoded_img = np.zeros_like(label_img,dtype='float')
    for coord,value in decoded_region_collection.points.items():
        decoded_img[coord] = value
    #decoded_img = np.rint(decoded_img).astype('uint8')
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

def psnr(img1,img2):
    mse = np.sum((img1 - img2)**2)
    if mse == 0:
        return(-1)
    mse /= img1.size
    mse = np.sqrt(mse)
    return(20*np.log10(255/mse))

def ssim(img1,img2):
    img1 = img1.astype('float64')
    img2 = img2.astype('float64')
    return(compare_ssim(img1,img2,dynamic_range=256))

def VSI(img1,img2):
    """Computes VSI of img1 vs. img2. Requires file VSI.m to be present in working directory"""
    
    from oct2py import octave
    fakergb1 = np.stack([img1,img1,img1],2).astype('float64')
    fakergb2 = np.stack([img2,img2,img2],2).astype('float64')
    fakergb1 /= 255
    fakergb2 /= 255
    octave.eval('pkg load image')
    vsi = octave.VSI(fakergb1,fakergb2)
    return(vsi)

def HaarPSI(img1,img2):
    """Computes HaarPSI of img1 vs. img2. Requires file HaarPSI.m to be present in working directory"""
    from oct2py import octave
    img1 = img1.astype('float64')
    img2 = img2.astype('float64')
    img1 /= 255
    img2 /= 255
    octave.eval('pkg load image')
    haarpsi = octave.HaarPSI(img1,img2)
    return(haarpsi)

class Image:
    def __init__(self):
        self.has_segmentation = False
        self.has_decoded_img = False
        self.method = None

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
                sigma = 2
            if 'min_size' in args.keys():
                min_size = args['min_size']
            else:
                min_size = 10
            self.label_img, self.label_pict = self.segmentation.felzenszwalb(scale,sigma,min_size)
            self.felz_scale,self.felz_sigma,self.felz_min_size = scale,sigma, min_size
        elif method == 'thresholded':
            threshold = args['threshold']
            sigma = args['sigma']
            self.label_img, self.label_pict = self.segmentation.thresholded(threshold,sigma)
        elif method == 'kmeans':
            nclusters = args['nclusters']
            self.label_img, self.label_pict = self.segmentation.kmeans(nclusters)
        self.has_segmentation = True
        
    def encode_rbepwt(self,levels, wavelet,path_type='easypath',euclidean_distance=True):
        self.method = 'rbepwt'
        self.rbewpt_path_type = path_type
        if not ispowerof2(self.img.size):
            raise Exception("Image size must be a power of 2")
        self.rbepwt_levels = levels
        self.rbepwt = Rbepwt(self,levels,wavelet,path_type=path_type)
        self.rbepwt.encode(euclidean_distance=euclidean_distance)

    def decode_rbepwt(self):
        self.decoded_region_collection = self.rbepwt.decode()
        self.decoded_img = np.zeros_like(self.img,dtype='float')
        for coord,value in self.decoded_region_collection.points.items():
            self.decoded_img[coord] = value
        #self.decoded_img = np.rint(self.decoded_img).astype('uint8')
        self.decoded_img[self.decoded_img > 255] = 255
        self.decoded_img[self.decoded_img < 0] = 0
        self.decoded_pict = Picture()
        self.decoded_pict.load_array(self.decoded_img)
        self.has_decoded_img = True
        
    def encode_dwt(self,levels,wavelet):
        self.method = 'dwt'
        if not ispowerof2(self.img.size):
            raise Exception("Image size must be a power of 2")
        self.dwt = Dwt(self,levels,wavelet)
        self.dwt_levels = levels
        self.dwt.encode()
        
    def decode_dwt(self):
        self.decoded_img = self.dwt.decode()
        self.decoded_img[self.decoded_img > 255] = 255
        self.decoded_img[self.decoded_img < 0] = 0
        self.decoded_pict = Picture()
        self.decoded_pict.load_array(self.decoded_img)
        self.has_decoded_img = True

    def encode_epwt(self,levels, wavelet):
        self.method = 'epwt'
        self.encode_rbepwt(levels,wavelet,'epwt-easypath')

    def decode_epwt(self):
        self.decode_rbepwt()


    def filter(self,sigma):
        self.filtered_img = np.zeros_like(self.img)
        for idx,region in self.rbepwt.region_collection_at_level[1]:
            base_points = []
            values = []
            for coord,value in region:
                base_points.append(coord)
                values.append(self.decoded_img[coord])
            tmpregion = Region(base_points,values)
            partial_img = skimage.filters.gaussian(tmpregion.get_enclosing_img(0),sigma)
            for coord,value in tmpregion:
                i = coord[0] - tmpregion.top_left[0]
                j = coord[1] - tmpregion.top_left[1]
                self.filtered_img[coord] = partial_img[i,j]
        self.filtered_pict = Picture()
        self.filtered_pict.load_array(self.filtered_img)
        #return(self.filtered_img)
        
    def psnr(self,filtered=True):
        """Returns PSNR (peak signal to noise ratio) of decoded image vs. original image"""

        if filtered:
            img = self.filtered_img
        else:
            img = self.decoded_img
        return(psnr(self.img,img))

    def ssim(self,filtered=True):
        """Retursn SSIM (Structural Similarity Index) of decoded image vs. original image"""

        if filtered:
            img = self.filtered_img
        else:
            img = self.decoded_img
        return(ssim(self.img,img))

    def vsi(self,filtered=True):
        """Returns VSI (Visual Saliency based Index) of decoded image vs. original image"""

        if filtered:
            img = self.filtered_img
        else:
            img = self.decoded_img
        return(VSI(self.img,img))

    def haarpsi(self,filtered=True):
        """Returns HaarPSI (Haar Perceptual Similarity Index) of decoded image vs. original image"""

        if filtered:
            img = self.filtered_img
        else:
            img = self.decoded_img
        return(HaarPSI(self.img,img))

    def error(self):
        print("PSNR: %f\nSSIM: %f\nVSI: %f\nHAARPSI: %f\n" %\
               (self.psnr(),self.ssim(),self.vsi(),self.haarpsi()))
    
    def nonzero_coefs(self):
        ncoefs = 0
        for level,arr in self.rbepwt.wavelet_details.items():
            ncoefs += arr.nonzero()[0].size
        ncoefs += self.rbepwt.region_collection_at_level[self.rbepwt_levels+1].values.nonzero()[0].size
        return(ncoefs)

    def nonzero_dwt_coefs(self):
        ncoefs = self.dwt.wavelet_coefs[0].nonzero()[0].size
        for tupl_coef in self.dwt.wavelet_coefs[1:]:
            for coef in tupl_coef:
                ncoefs += coef.nonzero()[0].size
        return(ncoefs)

    def threshold_coefs(self,ncoefs):
        if self.method in ['epwt','rbepwt']:
            self.rbepwt.threshold_coefs(ncoefs)
        elif self.method == 'dwt':
            self.dwt.threshold_coefs(ncoefs)
            
    def save_pickle(self,filepath):
        f = open(filepath,'wb')
        pickle.dump(self.__dict__,f,3)
        f.close()

    def load_pickle(self,filepath):
        f = open(filepath,'rb')
        tmpdict = pickle.load(f)
        f.close()
        self.__dict__.update(tmpdict)
        self.has_segmentation = True
        self.has_decoded_img = True
        if hasattr(self,'rbepwt'):
            self.rbepwt.img = self
        elif hasattr(self,'epwt'):
            self.rbepwt.img = self
        elif hasattr(self,'dwt'):
            self.dwt.img = self
        try:
            self.label_img
        except AttributeError:
            self.has_segmentation = False
        try:
            self.decoded_img
        except AttributeError:
            self.has_decoded_img = False
            
    def load_or_compute(self,imgpath,pickledpath,levels=12,wav='bior4.4'):
        try:
            self.load_pickle(pickledpath)
        except FileNotFoundError:
            self.read(imgpath)
            self.segment(scale=200,sigma=0.8,min_size=10)
            self.encode_rbepwt(levels,wav)
            self.save_pickle(pickledpath)

    def save(self,filepath):
        skimage.io.imsave(filepath,self.img)
            
    #def show(self,title='Original image',**args):
    #    self.pict.show(title=title,*args)

    def show(self,**other_args):
        if 'title' not in other_args.keys():
            other_args['title'] = 'Original Image'
        self.pict.show(**other_args)

    def show_decoded(self,**other_args):
        if 'title' not in other_args.keys():
            other_args['title'] = 'Decoded Image'
        self.decoded_pict.show(**other_args)

    def show_filtered(self,**other_args):
        if 'title' not in other_args.keys():
            other_args['title'] = 'Filtered Image'
        self.filtered_pict.show(**other_args)

    def save_decoded(self,filepath,**other_args):
        if 'title' not in other_args.keys():
            other_args['title'] = 'Decoded Image'
        self.decoded_pict.show(filepath=filepath,**other_args)
        
    def show_segmentation(self,**other_args):
        #self.label_pict.show(plt.cm.hsv)
        self.segmentation.show(**other_args)

    def save_segmentation(self,filepath,**other_args):
        if 'title' not in other_args.keys():
            other_args['title'] = None
        self.segmentation.save(filepath=filepath,**other_args)

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
            if self.array is not None:
                img = self.array/255.0
                skimage.io.imsave(filepath,img)
            elif self.mpl_fig is not None:
                fig.savefig(filepath,bbox_inches='tight',pad_inches=0.0)#,dpi='figure')
        
    def show(self,title=None,colormap=plt.cm.gray,filepath=None,border=False):
        """Shows self.array or self.mpl"""
        if self.array is not None:
            fig = plt.figure()
            axis = fig.gca()
            if border:
                axis.spines['top'].set_linewidth(2)
                axis.spines['right'].set_linewidth(2)
                axis.spines['bottom'].set_linewidth(2)
                axis.spines['left'].set_linewidth(2)
            plt.style.use('classic')
            plt.imshow(self.array, cmap=colormap, interpolation='none')
            plt.tick_params(
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                left='off',
                right='off',
                labelleft='off',
                labelbottom='off') 
            #plt.axis('off')
            if title is not None:
                plt.title(title)
            self.__save_or_show__(fig,filepath)
        elif self.mpl_fig is not None:
            ax = self.mpl_fig.gca()
            if title is not None:
                ax.set_title(title)
            self.__save_or_show__(self.mpl_fig,filepath)


class Segmentation:
    
    def __init__(self,image):
        self.img = image
        self.has_label_dict = False
        self.nlabels = -1

    def felzenszwalb(self,scale,sigma,min_size):
        self.label_img = felzenszwalb(self.img, scale=float(scale), sigma=float(sigma), min_size=int(min_size))
        self.compute_label_dict()
        self.nlabels = int(self.label_img.max()) + 1
        self.label_pict = Picture()
        self.label_pict.load_array(self.label_img)
        return(self.label_img,self.label_pict)

    def kmeans(self,nclusters):
        points = None
        for coord,value in np.ndenumerate(self.img):
            if points is not None:
                points = np.vstack((points,np.array([coord[0],coord[1],value])))
            else:
                points = np.array([coord[0],coord[1],value])
        km = skluster.KMeans(nclusters)
        km.fit(points)
        self.label_img = np.zeros_like(self.img)
        for idx,label in enumerate(km.labels_):
            coord = int(points[idx][0]),int(points[idx][1])
            self.label_img[coord] = label
        self.compute_label_dict()
        self.nlabels = nclusters
        self.label_pict = Picture()
        self.label_pict.load_array(self.label_img)
        return(self.label_img,self.label_pict)
        
    
    def thresholded(self,threshold,sigma=0.8):
        indexes = set(map(lambda x: x[0], np.ndenumerate(self.img)))
        fimg = skimage.filters.gaussian(self.img.astype('float64'),sigma=sigma)
        self.label_img = -np.ones_like(fimg)
        npoints = self.label_img.size
        label = 0
        avaiable_points = queue.Queue()
        avaiable_points.put((0,0))
        self.label_img[0,0] = label
        new_starting_points = queue.Queue()
        counter = 1
        while counter < npoints:
            if avaiable_points.empty():
                cand = new_starting_points.get()
                while self.label_img[cand] != -1:
                    cand = new_starting_points.get()
                avaiable_points.put(cand)
                label += 1
                self.label_img[cand] = label
            point = avaiable_points.get()
            for cand in set(neighborhood(point,1,hole=True)).intersection(indexes):
                diff = np.abs(fimg[cand] - fimg[point])
                if self.label_img[cand] == -1 and diff < threshold:
                    self.label_img[cand] = label
                    avaiable_points.put(cand)
                else:
                    new_starting_points.put(cand)
            counter += 1
        self.nlabels = int(self.label_img.max()) + 1
        self.label_pict = Picture()
        self.label_pict.load_array(self.label_img)
        return(self.label_img.astype('uint8'),self.label_pict)
                        
    def compute_label_dict(self):
        self.label_dict = {}
        for idx,label in np.ndenumerate(self.label_img):
            if label not in self.label_dict:
                self.label_dict[label] = Region([idx],[self.img[idx]])
            else:
                self.label_dict[label] += Region([idx],[self.img[idx]])
        self.has_label_dict = True
        return(self.label_dict)
                
    def estimate_perimeter(self):
        n,m = self.label_img.shape
        visited = set()
        self.borders = set()
        for coord,val in np.ndenumerate(self.label_img):
            neighbors = neighborhood(coord,1,'cross')
            neighbors.remove(coord)
            for neighbour in neighbors:
                i,j = neighbour
                if i < 0 or i >= n or j < 0 or j >= m:
                    continue
                couple = frozenset([coord,neighbour])
                if self.label_img[coord] != self.label_img[neighbour] and couple not in visited:
                    self.borders.add(couple)
                visited.add(couple)
        return(len(self.borders))

    def show(self,title=None,colorbar=True,border=False):
        fig = plt.figure()
        axis = fig.gca()
        plt.imshow(self.label_img,interpolation='none',cmap=plt.cm.plasma)
        if border:
            axis.spines['top'].set_linewidth(2)
            axis.spines['right'].set_linewidth(2)
            axis.spines['bottom'].set_linewidth(2)
            axis.spines['left'].set_linewidth(2)
        plt.tick_params(
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            left='off',
            right='off',
            labelleft='off',
            labelbottom='off')
        #plt.axis('off')
        if colorbar:
            plt.colorbar()
        self.pict = Picture()
        self.pict.load_mpl_fig(fig)
        self.pict.show(title)

    def save(self,filepath,title=None):
        fig = plt.figure()
        plt.imshow(self.label_img,interpolation='none',cmap=plt.cm.plasma)
        plt.axis('off')
        plt.colorbar()
        self.pict = Picture()
        self.pict.load_mpl_fig(fig)
        self.pict.show(title=title,filepath=filepath)
        
class Region:
    """Region of points, which can always be thought of as a path since points are ordered"""
    
    def __init__(self, base_points,  values=None,  compute_dict = True, avg_grad=None,):
        #if type(base_points) != type([]):
        #    raise Exception('The points to init the Path must be a list')
        if values is not None and len(base_points) != len(values):
            raise Exception('Input points and values must be of same length')
        self.points = {}
        self.base_points = tuple(base_points)
        #self.base_points = tuple(map(lambda x: tuple(x), base_points))
        self.trivial = False
        self.avg_gradient = avg_grad
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

    def pprint(self):
        print("Region top left: %s \t Region bottom right: %s \n Region points: \n %s" %\
              (self.top_left,self.bottom_right,self.base_points))
    
    def compute_avg_gradient(self,grad_matrix):
        avg_i, avg_j = 0,0
        for point in self.base_points:
            avg_i += grad_matrix[0][point]
            avg_j += grad_matrix[1][point]
        npoints = len(self.base_points)
        avg_i /= npoints
        avg_j /= npoints
        grad = np.array([avg_i,avg_j])
        grad /= np.linalg.norm(grad)
        #self.avg_gradient = (grad[1],grad[0])
        self.avg_gradient = grad
    
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

    def get_enclosing_img(self,fill_value=0):
        height = self.bottom_right[0] - self.top_left[0] + 1
        width = self.bottom_right[1] - self.top_left[1] + 1
        img = fill_value*np.ones((height,width))
        for coord,value in self:
            i = coord[0] - self.top_left[0]
            j = coord[1] - self.top_left[1]
            img[i,j] = value
        return(img)

    def grad_path(self,level,inplace=False,euclidean_distance=True):
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
        perp_grad_direc = rotate(self.avg_gradient, - np.pi/2)
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
                if euclidean_distance:
                    dist = np.linalg.norm(np.array(cur_point) - np.array(candidate))
                else:
                    dist = max(np.abs(cur_point[0]-candidate[0]),np.abs(cur_point[1]-candidate[1]))

                if  min_dist == None or dist < min_dist:
                    min_dist = dist
                    chosen_point  = candidate
                elif min_dist == dist:
                    #tmp_point = np.array(cur_point) + perp_grad_direc
                    v1 = np.array(chosen_point) - np.array(cur_point)
                    v2 = np.array(candidate) - np.array(cur_point)
                    v1 = v1/np.linalg.norm(v1)
                    v2 = v2/np.linalg.norm(v2)
                    sp1 = np.abs(np.dot(v1,perp_grad_direc))
                    sp2 = np.abs(np.dot(v2,perp_grad_direc))
                    if sp2 > sp1:
                        chosen_point = candidate
                    elif sp2 == sp1:
                        tmp_grad_direc = rotate(perp_grad_direc,- np.pi/2)
                        sp1 = np.abs(np.dot(v1,tmp_grad_direc))#
                        sp2 = np.abs(np.dot(v2,tmp_grad_direc))
                        if sp2 > sp1:
                            chosen_point = candidate
            if chosen_point != None:
                found += 1
                #new_path.add_point(chosen_point,self.points[chosen_point])
                new_base_points.append(chosen_point)
                new_values = np.append(new_values,self.points[chosen_point])
                avaiable_points.remove(chosen_point)
                last_direc = np.array(chosen_point) - np.array(cur_point)
                sp1 = np.dot(last_direc,perp_grad_direc)
                sp2 = np.dot(last_direc,-perp_grad_direc)
                if sp1 < sp2:
                    perp_grad_direc = -perp_grad_direc
                cur_point = chosen_point
                new_path_permutation.append(self.base_points.index(chosen_point))
                if not avaiable_points:
                    break
            else:
                print("This shouldn't happen! ", cur_point)
                raise Exception("Coulnd't find next point in path")
        if inplace:
            self.base_points = tuple(new_base_points)
            self.values = new_values
            #TODO: is the following needed?
            self.permutation = new_path_permutation
        new_path = Region(new_base_points, new_values)
        new_path.permutation = new_path_permutation
        new_path.avg_gradient = self.avg_gradient
        if _DEBUG:
            print("EASY PATH: permutation -- ", new_path.permutation)
        return(new_path)        
        
    def easy_path(self,level,inplace=False,epwt=False,euclidean_distance=True):
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
                neigh = neighborhood(cur_point,k,hole=True)
                candidate_points = avaiable_points.intersection(neigh)
                k+=1
            for candidate in candidate_points:
                if epwt:
                    dist = np.abs(self.points[cur_point] - self.points[candidate])
                elif euclidean_distance:
                    dist = np.linalg.norm(np.array(cur_point) - np.array(candidate))
                else:
                    dist = max(np.abs(cur_point[0]-candidate[0]),np.abs(cur_point[1]-candidate[1]))
                if  min_dist == None or dist < min_dist:
                    min_dist = dist
                    chosen_point  = candidate
                elif min_dist == dist:
                    v1 = np.array(chosen_point) - np.array(cur_point)
                    v2 = np.array(candidate) - np.array(cur_point)
                    v1 = v1/np.linalg.norm(v1)
                    v2 = v2/np.linalg.norm(v2)
                    sp1 = np.dot(v1,prefered_direc)
                    sp2 = np.dot(v2,prefered_direc)
                    if sp2 > sp1:
                        chosen_point = candidate
                    elif sp2 == sp1:
                        tmp_prefered_direc = rotate(prefered_direc,- np.pi/2)
                        sp1 = np.dot(v1,tmp_prefered_direc)#
                        sp2 = np.dot(v2,tmp_prefered_direc)
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
        #TODO: test these changes. encoding is a little faster...
        #new_values = []
        #new_points = []
        #l = len(self)
        #newl = np.floor(l/2) + (1-skip_first)*2*(l/2 - np.floor(l/2))
        #new_values = np.zeros(newl)
        #new_points = np.zeros(newl, dtype=[('x', 'i4'),('y', 'i4')])
        idx = 0
        for point,value in self:
            if idx % 2 == skip_first:
                new_region += Region([point],[value])
                #new_values.append(value)
                #new_points.append(point)
                #array_idx = np.floor(idx/2)
                #new_values[array_idx] = value
                #new_points[array_idx] = point
            idx += 1
        #new_region = Region(new_points,new_values)
        if len(new_region) == 0:
            new_region.no_values = True
        return(new_region)

    def show(self,show_path=False,title=None,point_size=5,px_value=False,fill=False,path_color='k',border_thickness = 0,border_color = 'black',alternate_markers=False,second_marker_color='blue'):
        pt_color = path_color
        rect_color = 'black'
        start_color = 'red'
        if px_value:
            fill = True
            
        fig = plt.figure()
        ax = fig.gca()
        ax.invert_yaxis()
        ax.set_xlim(self.top_left[1] - 1, self.bottom_right[1] + 1)
        ax.set_ylim(self.bottom_right[0] + 1,self.top_left[0] - 1)
        ax.set_aspect('equal')
        random_color = tuple([np.random.random() for i in range(3)])
        cur_point = self.base_points[0]
        i,j = cur_point
        plt.plot(j,i,'x',color=start_color,markeredgewidth=point_size/2,markersize=2*point_size)
        i -= 0.5
        j -= 0.5
        if px_value:
            if type(px_value) == type(0.1):
                col = str(px_value)
            else:
                col = str(min(1,self.points[cur_point]/255))
        else:
            col = rect_color
        ax.add_patch(patches.Rectangle((j,i),1,1,color=border_color,fill=fill))
        ax.add_patch(patches.Rectangle((j+border_thickness/2,i+border_thickness/2),1-border_thickness,1-border_thickness,color=col,fill=fill))
        ax.axis('off')
        iprev,jprev = i+0.5,j+0.5
        markers = ('x','o')
        for index,coord in enumerate(self.base_points[1:]):
            i,j = coord
            if show_path:
                x = jprev
                y = iprev
                dx = j-jprev
                dy = i-iprev
                plt.arrow(x,y,dx,dy,color=pt_color,length_includes_head=True,head_width=0.2)
            else:
                #plt.plot(j,i,'x',color=pt_color,markersize=point_size)
                plt.plot(j,i,'x',color=pt_color,markeredgewidth=point_size/2,markersize=2*point_size)
            if alternate_markers:
                col = [start_color,second_marker_color][(index+1)%2]
                plt.plot(j,i,markers[(index+1)%2],color=col,markeredgewidth=point_size/2,markersize=2*point_size)
            i -= 0.5
            j -= 0.5
            if px_value:
                if type(px_value) == type(0.1):
                    col = str(px_value)
                else:
                    col = str(min(1,self.points[coord]/255))
            else:
                col = rect_color
            ax.add_patch(patches.Rectangle((j,i),1,1,color=border_color,fill=fill))
            ax.add_patch(patches.Rectangle((j+border_thickness/2,i+border_thickness/2),1-border_thickness,1-border_thickness,color=col,fill=fill))
            iprev,jprev = i+0.5,j+0.5
        self.pict = Picture()
        self.pict.load_mpl_fig(fig)
        self.pict.show(title=title)

        
class RegionCollection:
    """Collection of Regions"""

    def __init__(self,  *regions):
        self.subregions = [] 
        self.nregions = 0
        self.region_lengths = []
        self.values = np.array([])
        self.base_points = []
        self.points = {}
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
                if _DEBUG and coord in points:
                    raise Exception("Conflicting coordinates in regions")
                else:
                    points.append(coord)
        self.add_regions(regions)
        
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

    def update(self):
        regions_copy = self.subregions
        self.__init__(*tuple(regions_copy))

    def add_region(self,region,copy_regions=True):
        for coord in region.base_points:
            if coord in self.base_points:
                raise Exception("Conflicting coordinates in regions")
        if copy_regions:
            newregion = copy.deepcopy(region) #TODO: do copy only once when adding multiple regions
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
        self.subregions.append(newregion)
        self.values = np.concatenate((self.values,newregion.values))
        self.base_points += newregion.base_points
        self.region_lengths.append(len(region))
        self.points = {**self.points, **region.points}
        self.nregions += 1
        self.no_regions = False

    def add_regions(self,regions):
        newregions = copy.deepcopy(tuple(regions))
        for region in newregions:
        #for region in regions:
            self.add_region(region,False)
        
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
            newregion.avg_gradient = subregion.avg_gradient
            newregion.values = values[prev_region_length:prev_region_length + len(newregion)]
            newregion.update_dict()
            prev_region_length += len(newregion)
            newregion.generating_permutation = subregion.permutation
            new_region_collection.add_region(newregion)
        #print("\n\n",len(new_region_collection.points))
        return(new_region_collection)

    def expand(self,values, upper_region_collection, wavelet): #TODO: why is wavelet needed as argument??
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
                if subr.avg_gradient is not None:
                    length = 20
                    to_x,to_y = length*subr.avg_gradient[0],length*subr.avg_gradient[1]
                    plt.arrow(xp,yp,to_x,to_y,color=random_color,length_includes_head=True,head_width=2)
                    #plt.annotate("",xy=(xp,yp), xytext=(to_x,to_y), arrowprops=dict(arrowstyle="->",color=random_color))
                plt.plot([xp],[yp], 'o', ms=point_size,mew=5,color=random_color)
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
    def __init__(self, img, levels, wavelet, path_type='easypath'):
        if 2**levels > img.size:
            raise Exception('2^levels must be smaller or equal to the number of pixels in the image')
        if type(img).__name__ != type(Image()).__name__:
            raise Exception('First argument must be an Image instance')
        self.img = img
        self.levels = levels
        self.has_encoding = False
        self.wavelet = wavelet
        self.path_type = path_type

    def wavelet_coefs_dict(self):
        """Returns a dictionary with the wavelet detail coefficients for every level plus the wavelet approximation coefficients at the end"""

        out_dict = self.wavelet_details
        out_dict[self.levels+1] = self.region_collection_at_level[self.levels+1].values
        return(out_dict)

    def encode(self,onlypaths=False,euclidean_distance=True):
        wavelet=self.wavelet
        if not self.img.has_segmentation and self.path_type != 'epwt-easypath':
            print('Segmenting image with default parameters...')
            self.img.segment()
        if self.path_type == 'epwt-easypath':
            base_points,values = zip(*[(coord,value) for coord,value in np.ndenumerate(self.img.img)])
            self.region_collection_at_level = {1: RegionCollection(Region(base_points,values))}
        else:
            regions = self.img.segmentation.label_dict.values()
            self.region_collection_at_level = {1: RegionCollection(*tuple(regions))}
        if self.path_type == 'gradpath':
            grad_matrix = np.gradient(self.img.img.astype('float64'))
            for key,r in self.region_collection_at_level[1]:
                r.compute_avg_gradient(grad_matrix)
        #self.region_collection_at_level = [None,RegionCollection(*tuple(regions))]
        if onlypaths:
            self.region_collection_at_level[1].values = np.zeros(len(regions))
        self.wavelet_details = {}
        for level in range(1,self.levels+1):
            level_length = 0
            paths_at_level = []
            cur_region_collection = self.region_collection_at_level[level]
            for key, subregion in cur_region_collection:
                level_length += len(subregion)
                if self.path_type == 'easypath':
                    paths_at_level.append(subregion.easy_path(level,inplace=True,euclidean_distance=euclidean_distance))
                elif self.path_type == 'gradpath':
                    paths_at_level.append(subregion.grad_path(level,inplace=True,euclidean_distance=euclidean_distance))
                elif self.path_type == 'epwt-easypath':
                    paths_at_level.append(subregion.easy_path(level,inplace=True,epwt=True))
            cur_region_collection = RegionCollection(*paths_at_level)
            if onlypaths:
                wlen = len(cur_region_collection.values)/2
                wapprox,wdetail = np.zeros(wlen),np.zeros(wlen)
            else:
                wapprox,wdetail = pywt.dwt(cur_region_collection.values, wavelet,'periodization')
            #print("wdetail size: %d" % wdetail.size)
            self.wavelet_details[level] = wdetail
            self.region_collection_at_level[level+1] = cur_region_collection.reduce(wapprox)
            print("\n-- ENCODING: finished working on level %d" % level)
            if _DEBUG:
                for key, subr in self.region_collection_at_level[level]:
                    print("ENCODING: subregion %s has base points %s" % (key,subr.base_points))
                    print("ENCODING: subregion %s has base values %s" % (key,subr.values))
                #print("ENCODING: self.region_collection_at_level[level].values %s" % self.region_collection_at_level[level].values)
                print("ENCODING: self.wavelet_details[level] %s" % self.wavelet_details[level])
                print("ENCODING: self.region_collection_at_level[level+1].values %s" % self.region_collection_at_level[level+1].values)
        self.has_encoding = True

    def decode(self):
        wavelet=self.wavelet
        if not self.has_encoding:
            raise Exception("There is no saved encoding to decode")
        nonzerocoefs = self.region_collection_at_level[self.levels+1].values.nonzero()[0].size
        cur_region_collection = self.region_collection_at_level[self.levels+1]
        values = self.region_collection_at_level[self.levels+1].values
        for level in range(self.levels,0, -1):
            #ipdb.set_trace()
            cur_region_collection = self.region_collection_at_level[level+1]
            cur_region_collection.values = values #APPLY PERMUTATION FIRST
            wdetail,wapprox = self.wavelet_details[level], cur_region_collection.values
            nonzerocoefs += wdetail.nonzero()[0].size
            values = pywt.idwt(wapprox, wdetail, wavelet,'periodization')
            upper_region_collection = self.region_collection_at_level[level]
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
        wapprox_rc = self.region_collection_at_level[self.levels + 1]
        wav_approx = wapprox_rc.values
        lev_length = len(wav_approx)
        flat_coefs = np.append(flat_coefs,np.stack(((self.levels+1)*np.ones(lev_length),wav_approx)),1)
        sorted_idx = np.argsort(np.abs(flat_coefs[1,:]))
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
            lev_length = len(self.region_collection_at_level[lev + 1].base_points)
            for k in range(lev_length):
                self.wavelet_details[lev][k] = flat_thresholded_coefs[1,prev_len + k]
            prev_len += lev_length
        for k in range(len(wapprox_rc.base_points)):
            self.region_collection_at_level[self.levels +1].values[k] = flat_thresholded_coefs[1,prev_len +k]
            
    def threshold_coeffs_by_value(self,threshold,threshold_type='hard'): #TODO: never tested this
        for level in range(2,self.levels+2):
            if threshold_type == 'hard':
                idx = self.wavelet_details[level] > threshold
                self.wavelet_details[level] = self.wavelet_details[level][idx]

    def threshold_by_percentage(self,perc):
        """Keeps only perc proportion of coefficients for each region"""

        #encode new data structure
        region_coefs_dict = {}
        for level,coefs in self.wavelet_details.items():
            prev_len = 0
            for key, region in enumerate(self.region_collection_at_level[level+1].subregions):
                region_length = len(region)
                region_coefs = coefs[prev_len: prev_len + region_length]
                if key not in region_coefs_dict.keys():
                    region_coefs_dict[key] = np.stack((level*np.ones(region_length),region_coefs),0)
                else:
                    region_coefs_dict[key] = np.append(region_coefs_dict[key],\
                                            np.stack((level*np.ones(region_length),region_coefs),0),1)
                prev_len += region_length
        level = self.levels + 1
        wapprox = self.region_collection_at_level[level].values
        prev_len = 0
        for key,region in enumerate(self.region_collection_at_level[level].subregions):
            region_length = len(region)
            region_coefs = wapprox[prev_len: prev_len + region_length]
            if key not in region_coefs_dict.keys():
                region_coefs_dict[key] = np.stack((level*np.ones(region_length),region_coefs),0)
            else:
                region_coefs_dict[key] = np.append(region_coefs_dict[key],\
                                        np.stack((level*np.ones(region_length),region_coefs),0),1)
            prev_len += region_length
        #threshold coefficients:
        thresholded_region_coefs_dict = {}
        sorted_idx = {}
        for key,coefs in region_coefs_dict.items():
            sorted_idx[key] = np.flipud(np.argsort(np.abs(coefs[1,:])))
            thresholded_region_coefs_dict[key] = np.zeros_like(coefs,dtype='float')
            thresholded_region_coefs_dict[key][0,:] = coefs[0,:]
            ncoefs = int(min(myround(perc*coefs.shape[1]),coefs.shape[1]))
            for i in range(ncoefs):
                idx = sorted_idx[key][i]
                thresholded_region_coefs_dict[key][1,idx] = coefs[1,idx]
        #decode data structure (i.e. copy over thresholded coefficients)
        prev_len = {}
        for key, region in enumerate(self.region_collection_at_level[1].subregions):
            prev_len[key] = 0
        wdetails = {}
        for level in range(1,self.levels+1):
            for key, region in enumerate(self.region_collection_at_level[level+1].subregions):
                #ipdb.set_trace()
                region_length = len(region)
                try:
                    wdetails[level] = np.append(wdetails[level],thresholded_region_coefs_dict[key]\
                                           [1,prev_len[key]: prev_len[key] + region_length])
                except KeyError:
                    wdetails[level] = thresholded_region_coefs_dict[key]\
                                           [1,prev_len[key]: prev_len[key] + region_length]
                prev_len[key] += region_length
        level = self.levels + 1
        #prev_len = 0
        #ipdb.set_trace()
        for key,region in enumerate(self.region_collection_at_level[level].subregions):
            region_length = len(region)
            region_coefs = thresholded_region_coefs_dict[key][1,prev_len[key]: prev_len[key] + region_length]
            #loca = self.region_collection_at_level
            #ipdb.set_trace()
            try:
                wapprox = np.append(wapprox,region_coefs)
            except NameError:
                wapprox = region_coefs
        for level in range(1,self.levels+1):
            self.wavelet_details[level] = wdetails[level]
        self.region_collection_at_level[self.levels+1].values = wapprox
        self.region_collection_at_level[self.levels+1].update()
        #for level in range(1,self.levels+1):
        #    self.region_collection_at_level[level].update()
            
        
    def flat_wavelet(self):
        """Returns an array with all wavelet coefficients sequentially"""

        ret = np.array([])
        for lev in range(1,self.levels+1):
            wavelet_at_level = self.wavelet_details[lev]
            #print(lev,wavelet_at_level.size)
            ret = np.append(ret,wavelet_at_level)
        ret = np.append(ret,self.region_collection_at_level[self.levels+1].values)
        return(ret)

    def show_wavelets(self):
        fig = plt.figure()
        nplots = self.levels + 1
        for lev in range(1,self.levels +1):
            vec = self.wavelet_details[lev]
            plt.subplot(nplots,1,lev)
            plt.plot(vec)
        plt.subplot(nplots,1,self.levels+1)
        plt.plot(self.region_collection_at_level[self.levels+1].values)
        self.wavs_pict = Picture()
        self.wavs_pict.load_mpl_fig(fig)
        self.wavs_pict.show()
    
    def show_wavelets_at_levels(self,levels=None,show_approx=True):
        if levels == None:
            levels = range(1,self.levels+1)
        for level in levels:
            self.show_wavelet_detail_at_level(level)
        if show_approx==True:
            self.region_collection_at_level[self.levels+1].show_values('Wavelet approximation coefficients')
            
    def show_wavelet_detail_at_level(self,level):
        if level in self.wavelet_details:
            fig = plt.figure()
            plt.title('Wavelet detail coefficients at level %d ' % level)
            plt.plot(self.wavelet_details[level])
            #for key,subr in self.region_collection_at_level[level]:
            #    print(key)
            #    miny,maxy = min(self.wavelet_details[level]),max(self.wavelet_details[level])
            #    x = self.region_collection_at_level[level].region_lengths[key]
            #    #plt.plot([x,x],[miny,maxy],'r') #TODO: is this correct? should we use x/2 instead?
            self.wd_at_level = Picture()
            self.wd_at_level.load_mpl_fig(fig)
            self.wd_at_level.show()
    
    def show(self,levels=None,point_size=5):
        if levels == None:
            levels = range(1,self.levels+1)
        for lev in levels:
            if lev in self.region_collection_at_level.keys():
                self.region_collection_at_level[lev].show('Region collection at level %d' % lev)

        
class Dwt: 
    def __init__(self, img, levels, wavelet):
        if 2**levels > img.size:
            raise Exception('2^levels must be smaller or equal to the number of pixels in the image')
        if type(img).__name__ != type(Image()).__name__:
            raise Exception('First argument must be an Image instance')
        self.img = img
        self.levels = levels
        self.has_encoding = False
        self.wavelet = wavelet

    def encode(self):
        self.wavelet_coefs = pywt.wavedec2(self.img.img, self.wavelet, level=self.levels, mode='periodization')

    def decode(self):
        return(pywt.waverec2(self.wavelet_coefs,self.wavelet,mode='periodization'))

    def threshold_coefs(self,ncoefs):
        N = self.img.size
        flat_coefs = self.wavelet_coefs[0].flatten()
        last_idx = len(flat_coefs)
        for wdetail in self.wavelet_coefs[1:]:
            horiz_detail,vert_detail,diag_detail = wdetail
            horiz_detail = horiz_detail.flatten()
            vert_detail = vert_detail.flatten()
            diag_detail = diag_detail.flatten()
            flatted = np.concatenate((horiz_detail,vert_detail,diag_detail))
            flat_coefs = np.append(flat_coefs,flatted)
        thresholded_coefs = np.zeros_like(flat_coefs)
        sorted_idx = np.argsort(np.abs(flat_coefs))
        count = 0
        for idx in reversed(sorted_idx):
            thresholded_coefs[idx] = flat_coefs[idx]
            count += 1
            if count == ncoefs:
                break
        length = self.wavelet_coefs[0].shape[0]
        size = length**2
        new_approx = thresholded_coefs[:size].reshape(length,length)
        last_idx = size
        new_coefs = [new_approx]
        for level in range(self.levels):
            new_hdetail = thresholded_coefs[last_idx:last_idx + size].reshape(length,length)
            new_vdetail = thresholded_coefs[last_idx + size:last_idx + 2*size].reshape(length,length)
            new_ddetail = thresholded_coefs[last_idx + 2*size:last_idx + 3*size].reshape(length,length)
            new_coefs.append((new_hdetail,new_vdetail,new_ddetail))
            last_idx += 3*size
            length *= 2
            size = length**2
        self.wavelet_coefs = new_coefs


