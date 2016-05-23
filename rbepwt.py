#!/usr/bin/env python

class Image:
    
    def __init__(self):
        pass

    def read(self,filepath):
        pass

    def segment(self,method):
        self.segmentation = Segmentation(self)

    def rbepwt(self):
        pass

    def irbepwt(self):
        #return(reconstructed_image)
        pass

    def threshold_coeffs(self,threshold_type):
        pass

    def psnr(self):
        pass
    
class Segmentation:

    def __init__(self,image):
        pass

    def felzenszwalb(self,parameters):
        pass

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
