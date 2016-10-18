#!/Users/renato/anaconda/bin/python

from draw_paths import grad_and_norm, plot_array
import sys
import scipy.io
import skimage.segmentation as skiseg
from skimage.io import imread
import numpy as np
import skimage.morphology as skimorph
import skimage.filters as skifil

def usage():
    string = "USAGE: %s imgpath scale sigma min_size" % sys.argv[0]

    print(string)

def main(imgpath, scale, sigma, min_size):
    img = imread(imgpath)

    #if int(grad) == 1:
    #    #img = grad_and_norm(img)[1]
    #        rect = skimorph.rectangle
    #        neighbourhood = rect(2,2)
    #        img = skifil.rank.gradient(img,neighbourhood)
    print("args:  ", imgpath, scale, sigma, min_size)
    label_img = skiseg.felzenszwalb(img, scale=float(scale), sigma=float(sigma), min_size=int(min_size))

    #plot_array(img)
    #plot_array(label_img)
    #save as matlab
    scipy.io.savemat(imgpath+'.mat',dict(label_img=label_img))

if __name__ == '__main__':
    nargs = len(sys.argv)
    if nargs < 5:
        usage()
        exit(1)
    else:
        imgpath, scale, sigma, min_size = sys.argv[1:]
        main(imgpath, scale, sigma, min_size)
        
    
