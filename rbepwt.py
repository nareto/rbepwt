#!/usr/bin/env python
import ipdb
import copy
import PIL
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pywt
from skimage.segmentation import felzenszwalb 

#def concat_two_paths(instpath1,instpath2):
#    points = []
#    values = []
#    for idx,val in instpath1.points.items():
#        p,v = val
#        points.append(p)
#        values.append(v)
#    for idx,val in instpath2.points.items():
#        p,v = val
#        points.append(p)
#        values.append(v)
#    newinst = Path(points,values)
#    return(newinst)
#
#def concat_paths(*paths):
#    prev = Path([])
#    for cur in paths:
#        prev = concat_two_paths(prev,cur)
#    return(prev)

def rotate(vector,theta):
    """Rotates a 2D vector counterclockwise by theta"""
    
    matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return(np.dot(matrix,vector))

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
        
    def encode_rbepwt(self,levels=2):
        #if type(self) != type(Image()):
        #    ipdb.set_trace()
        self.rbepwt = Rbepwt(self,levels)
        self.rbepwt.encode()

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

class Segmentation:
    
    def __init__(self,image):
        self.img = image
        self.has_label_dict = False

    def felzenszwalb(self,scale=200,sigma=0.8,min_size=10):
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
        plt.imshow(self.label_img,interpolation='none')
        self.pict = Picture()
        self.pict.load_mpl_fig(fig)
        self.pict.show()

class Region:
    """Region of points, which can always be thought of as a path since points are ordered"""
    
    def __init__(self, base_points, values=None, compute_dict = True):
        #if type(base_points) != type([]):
        #    raise Exception('The points to init the Path must be a list')
        if values != None and len(base_points) != len(values):
            raise Exception('Input points and values must be of same length')
        self.points = {}
        self.base_points = tuple(base_points)
        if values == None:
            self.values = np.array(len(base_points)*[None])
            self.no_values = True
        else:
            self.values = np.array(values)
            self.no_values = False
        if compute_dict:
            self.__init_dict_and_extreme_values__()
        
    def __init_dict_and_extreme_values__(self):
        self.trivial = False
        try:
            top_left = self.base_points[0]
            bottom_right = self.base_points[0]
        except IndexError:
            self.top_left,self.bottom_right = None,None
            self.trivial = True
        if not self.trivial:
            bp = self.base_points
            for i in range(len(bp)):
                if self.no_values:
                    self.points[bp[i]] = None
                else:
                    self.points[bp[i]] = self.values[i]
                row,col = bp[i]
                top_left = min(row,top_left[0]),min(col,top_left[1])
                bottom_right = max(row,bottom_right[0]),max(col,bottom_right[1])
            self.top_left = top_left
            self.bottom_right= bottom_right
                

    def __getitem__(self, key):
        return((self.base_points[key],self.values[key]))

    def __add__(self,region):
        basepoints = self.base_points + region.base_points
        values = np.concatenate((self.values,region.values))
        newregion = Region(basepoints,values,False)
        newregion.points = {**self.points, **region.points} #syntax valid from python3.5. see http://stackoverflow.com/questions/38987
        tl1,br1,tl2,br2 = self.top_left, self.bottom_right, region.top_left, region.bottom_right
        if tl1 == None:
            newregion.top_left = tl2
            newregion.bottom_right = br2
        elif tl2 == None:
            newregion.top_left = tl1
            newregion.bottom_right = br1
        else:
            newregion.top_left = min(tl1[0],tl2[0]),min(tl1[1],tl2[1])
            newregion.bottom_right = max(tl1[0],tl2[0]),max(tl1[1],tl2[1])
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

    def add_point(self,coord,value):
        lbp = list(self.base_points)
        lbp.append(coord)
        #ipdb.set_trace()
        self.base_points = tuple(lbp)
        lv = list(self.values)
        lv.append(value)
        self.values = tuple(lv)
        self.points[coord] = value
        i,j = coord
        self.top_left = min(i,self.top_left[0]),min(j,self.top_left[1])
        self.bottom_right_ = max(i,self.bottom_right[0]),max(j,self.bottom_right[1])
    
    def lazy_path(self,reorder_self=False): #TODO: update self.permutation - needed for decoding
        start_point = self.top_left
        if len(self) <= 1:
            return(self)
        #new_path = Region([start_point],[self.points[start_point]])
        bp = tuple(filter(lambda point: point != start_point,self.base_points))
        avaiable_points = set(bp)
        if len(bp) == 0:
            return(Region([start_point],[self.points[start_point]]))
        new_path_permutation = []
        new_base_points = [start_point]
        new_values = np.array(self.points[start_point])
        cur_point = start_point
        found = 0
        prefered_direc = np.array((0,1))
        while cur_point != None:
            chosen_point = None
            min_dist = None
            for candidate in avaiable_points:
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
                print("This shouldn't happen! ", cur_point)
        #print(len([start_point]+new_base_points), len( [self.points[start_point]] + new_values))
        new_path = Region(new_base_points, new_values)
        new_path.permutation = new_path_permutation
        print("permutation -- ", new_path.permutation)
        return(new_path)
            
    def reduce_points(self,skip_first=False):
        if len(self) <= 1:
            return(Region([],[]))
        new_region = Region([],[])
        idx = 0
        for point,value in self:
            if idx % 2 == skip_first:
                new_region += Region([point],[value])
            idx += 1
            #print(point,value)
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
        self.subregions = {}
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
        self.top_left = min(self.top_left[0],region.top_left[0]), min(self.top_left[1],region.top_left[1])
        self.bottom_right = max(self.bottom_right[0],region.bottom_right[0]), max(self.bottom_right[1],region.bottom_right[1])
        self.subregions[self.nregions] = newregion
        self.values = np.concatenate((self.values,newregion.values))
        self.base_points += newregion.base_points
        self.region_lengths.append(len(region))
        self.points = {**self.points, **region.points}
        self.nregions += 1

    def wavelet_and_reduce(self,wavelet='db1'):
        """Returns wavelet details for current level and a new region collection for the next"""
        
        new_region_collection = RegionCollection()
        wapprox,wdetail = pywt.dwt(self.values, wavelet)
        skipped_prev,prev_had_odd_length = False, False
        prev_region_length = 0
        for key,subregion in self.subregions.items():
            if skipped_prev + prev_had_odd_length == True:
                skip_first = True
            else:
                skip_first = False
            skipped_prev = skip_first
            prev_had_odd_length = len(subregion) % 2
            newregion = subregion.reduce_points(skip_first)
            newregion.values = wapprox[prev_region_length:prev_region_length + len(newregion)]
            prev_region_length += len(newregion)
            new_region_collection.add_region(newregion)
        return(wdetail,new_region_collection)

    def expand_and_wavelet(self,wavelet='db1'):
        """Returns wavelet approximation coefficients for current level and new region collection for the previous"""
        #TODO: this needs to be aware of the permutation...
        pass

    def show(self,level=None,point_size=5):
        fig = plt.figure()
        if level != None:
            plt.title('Region collection at level %d' % level)
        n,m = self.bottom_right[0] - self.top_left[0], self.bottom_right[1] - self.top_left[1]
        #paths = self.paths[lev]
        #for key,subregion in self:
        i,j = self.base_points[0]
        xp,yp = j,n-i
        random_color = tuple([np.random.random() for i in range(3)])
        plt.plot([xp],[yp], '+', ms=2*point_size,mew=10,color=random_color)
        for p in self.base_points[1:]:
            #random_color = tuple([np.random.random() for i in range(3)])
            offset=0.3
            #for p in subregion.base_points[1:]:
            i,j = p
            x,y = j, n-i
            if max(abs(x-xp), abs(y-yp)) < -11:
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
            
class Rbepwt:
    def __init__(self, img, levels=2):
        #if type(img) != type(Image()):
        #    raise Exception('First argument must be an Image instance')
        self.img = img
        self.levels = levels

    def encode(self,wavelet='db1'):
        if not self.img.has_segmentation:
            self.img.segment()
        regions = self.img.segmentation.label_dict.values()
        self.region_collection_dict = {0: RegionCollection(*tuple(regions))}
        self.wavelet_details = {}
        for level in range(1,self.levels+1):
            level_length = 0
            paths_at_level = []
            for key, subregion in self.region_collection_dict[level-1]:
                level_length += len(subregion)
                paths_at_level.append(subregion.lazy_path())
            tmp_region_collection = RegionCollection(*paths_at_level)
            self.wavelet_details[level], self.region_collection_dict[level] = tmp_region_collection.wavelet_and_reduce(wavelet)
            print('Finished working on level %d with %d points'  %(level, level_length))
            

    def threshold_coeffs(self,threshold,threshold_type='hard'): #TODO: never tested this
        for level in range(1,self.levels+1):
            if threshold_type == 'hard':
                idx = self.wavelet_details[level] < threshold
                self.wavelet_details[level] = self.wavelet_details[level][idx]

    def show_wavelet(self,level):
        fig = plt.figure()
        plt.title('Wavelet detail coefficients at level %d ' % level)
        plt.plot(self.wavelet_details[level])
        for key,subr in self.region_collection_dict[level]:
            print(key)
            miny,maxy = min(self.wavelet_details[level]),max(self.wavelet_details[level])
            x = self.region_collection_dict[level].region_lengths[key]
            #plt.plot([x,x],[miny,maxy],'r') #TODO: is this correct? should we use x/2 instead?
        self.pict = Picture()
        self.pict.load_mpl_fig(fig)
        self.pict.show()
    
    def show(self,levels=None,point_size=5):
        if levels == None:
            levels = self.levels
        for lev in range(1,levels+1):
            self.region_collection_dict[lev].show(lev)
        
            
