import sys,os
import rbepwt
import numpy as np

def main(filepath,wavelet='bior4.4',levels=16,path_type='easypath',paths_only_at_first_level=False):
    filename,ext = os.path.splitext(filepath)
    img = filename.split('/')[-1]
    bonustring = ''
    if paths_only_at_first_level:
        bonustring = '-ponly_first_level'
    pickled_string='../pickled/'+img+bonustring+'-%s-%s-%dlevels'%(path_type,wavelet,levels)
    print("Will save encoded pickle as: %s" % pickled_string)
    
    i = rbepwt.Image()
    i.read(filepath)
    print("Working on segementation of image %s ..." % filepath)
    i.segment(scale=200,sigma=2,min_size=10)
    #i.segment(method='kmeans',nclusters=30)
    print("Encoding image %s ..." % filepath)
    i.encode_rbepwt(levels,wavelet,path_type=path_type,paths_first_level = paths_only_at_first_level)
    i.save_pickle(pickled_string)

if __name__ == '__main__':
    main(sys.argv[1])
