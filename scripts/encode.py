import sys,os
import rbepwt
import numpy as np

def main(filepath,wavelet='bior4.4',levels=12,path_type='easypath'):
    filename,ext = os.path.splitext(filepath)
    img = filename.split('/')[-1]
    pickled_string='pickled/'+img+'-%s-%s-%dlevels'%(path_type,wavelet,levels)

    i = rbepwt.Image()
    i.read(filepath)
    print("Working on segemntation of image %s ..." % filepath)
    i.segment(scale=200,sigma=2,min_size=10)
    print("Encoding image %s ..." % filepath)
    i.encode_rbepwt(levels,wavelet,path_type=path_type)
    i.save(pickled_string)

if __name__ == '__main__':
    main(sys.argv[1])
