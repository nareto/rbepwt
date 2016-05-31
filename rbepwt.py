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
        
    def compute_rbepwt(self,levels=2):
        #if type(self) != type(Image()):
        #    ipdb.set_trace()
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
        #if type(img) != type(Image()):
        #    raise Exception('First argument must be an Image instance')
        self.img = img
        self.levels = levels

    def compute(self,wavelet='db1'):
        #self.__init_path_data_structure__()
        if not self.img.has_segmentation:
            self.img.segment()
        self.regions={1: self.img.segmentation.label_dict}
        self.paths = {}
        for level in range(1,self.levels+1):
            skipped_prev,prev_had_odd_length = False, False
            #self.paths[level] = {}
            level_length = 0
            #values_at_level = []
            #for label,region in self.img.segmentation.label_dict.items():
            paths_at_level = []
            for label, region in self.regions[level]: #TODO: use self.paths[level]
                if level == 1:
                    path = region.lazy_path()
                    #self.paths[level] = path
                else: #TODO: this else shouldn't be needed
                    if skipped_prev + prev_had_odd_length == True:
                        skip_first = True
                    else:
                        skip_first = False
                    #path = self.paths[level - 1][label].reduce_points(skip_first)
                    #path = path.lazy_path()
                    path = region.lazy_path()
                    #self.paths[level] += path
                    prev_had_odd_length = len(path) % 2
                    skipped_prev = skip_first
                paths_at_level += [path]
                #values_at_level += path.values
                level_length += len(path)
            self.paths[level] = paths_at_level[0].merge(paths_at_level[1:])
            wapprox,wdetail = pywt.dwt(self.paths[level].values, wavelet)
            
            print("Level %d has %d points" % (level,level_length))
        
    def threshold_coeffs(self,threshold):
        pass

    def show(self,levels=None,point_size=5):
        if levels == None:
            levels = self.levels
        for lev in range(1,levels+1):
            fig = plt.figure()
            plt.title('Paths at level %d' % lev)
            n,m = self.img.shape
            paths = self.paths[lev]
            for label,path in paths.items():
                random_color = tuple([np.random.random() for i in range(3)])
                offset=0.3
                i,j = path.base_points[0]
                xp,yp = j,n-i
                plt.plot([xp],[yp], '+', ms=2*point_size,mew=10,color=random_color)
                for p in path.base_points[1:]:
                    i,j = p
                    x,y = j, n-i
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
            self.values = tuple(len(base_points)*[None])
        else:
            self.values = tuple(values)
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
                if self.values == None:
                    self.points[bp[i]] = None
                else:
                    self.points[bp[i]] = self.values[i]
                row,col = bp[i]
                if row < top_left[0] or (row == top_left[0] and col < top_left[0]):
                    top_left = bp[i]
                if row > bottom_right[0] or (row == bottom_right[0] and col > bottom_right[0]):
                    bottom_right = bp[i]
            self.top_left = top_left
            self.bottom_right= bottom_right
                

    def __getitem__(self, key):
        return((self.base_points[key],self.values[key]))

    def __add__(self,region):
        basepoints = self.base_points + region.base_points
        values = self.values + region.values
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
            if tl1[0] < tl2[0] or (tl1[0] == tl2[0] and tl1[1] < tl2[1]):
                newregion.top_left = tl1
            else:
                newregion.top_left = tl2
            if br1[0] > br2[0] or (br1[0] == br2[0] and br1[1] > br2[1]):
                newregion.bottom_right = br1
            else:
                newregion.bottom_right = br2
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

    def lazy_path(self):
        start_point = self.top_left
        if len(self) <= 1:
            return(self)
        new_path = Region([start_point],[self.points[start_point]])
        bp = tuple(filter(lambda point: point != start_point,self.base_points))
        avaiable_points = set(bp)
        if len(bp) == 0:
            return(new_path)
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
                new_path += Region([chosen_point],[self.points[chosen_point]])
                avaiable_points.remove(chosen_point)
                prefered_direc = np.array(chosen_point) - np.array(cur_point)
                cur_point = chosen_point
                if not avaiable_points:
                    break
            else:
                print("This shouldn't happen! ", cur_point)
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
    
    def wavelet_transform(self,wavelet):
        pass

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

        
class ComplexRegion(Region):
    """Collection of Regions"""
    
    def __init__(self,  *regions, copy_regions=True):
        self.subregions = {}
        self.nregions = 0
        self.region_lengths = []
        self.values = []
        self.base_points = []
        self.points = {}
        self.copy_regions = copy_regions
        points = []
        for r in regions:
            for coord in r.base_points:
                if coord in points:
                    raise Exception("Conflicting coordinates in regions")
                else:
                    points.append(coord)
        for r in regions:
            self.add_region(r)
            self.values += r.values
            self.base_points += r.base_points
            self.region_lengths += [len(r)]
            self.points = {**self.points, **r.points}

    def __len__(self):
        return(self.nregions)
            
    def add_region(self,region):
        for coord in region.base_points:
            if coord in self.base_points:
                raise Exception("Conflicting coordinates in regions")
        if self.copy_regions:
            newregion = copy.deepcopy(region)
        else:
            newregion = region
        self.subregions[self.nregions] = newregion
        self.values += newregion.values
        self.base_points += newregion.base_points
        self.region_lengths.append(len(region))
        self.nregions += 1
    
    def __getitem__(self,key):
        for key,subr in self.subregions.items():
            if key in subr.points.keys():
                return(subr[key])
        
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
            
