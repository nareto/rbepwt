import sys,os
import rbepwt
import numpy as np

def main(filepath,wavelet='bior4.4',levels=16,path_type='easypath'):
    filename,ext = os.path.splitext(filepath)
    img = filename.split('/')[-1]
    pickled_string='../pickled/'+img+'-%s-%s-%dlevels'%(path_type,wavelet,levels)
    #print(pickled_string)
    
    i = rbepwt.Image()
    i.read(filepath)
    print("Working on segemntation of image %s ..." % filepath)
    #i.segment(scale=200,sigma=2,min_size=10)
    i.segment(method='kmeans',nclusters=30)
    print("Encoding image %s ..." % filepath)
    i.encode_rbepwt(levels,wavelet,path_type=path_type)
    i.save_pickle(pickled_string)

if __name__ == '__main__':
    main(sys.argv[1])
