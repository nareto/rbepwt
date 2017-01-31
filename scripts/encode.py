import sys,os
import rbepwt
import numpy as np
import time

pickleddir = '../pickled/'
decodedpickleddir = '../decoded_pickles-euclidean/'

def encode(filepath,wavelet='bior4.4',levels=16,path_type='easypath',paths_only_at_first_level=False,save=True):
    filename,ext = os.path.splitext(filepath)
    img = filename.split('/')[-1]
    bonustring = ''
    if paths_only_at_first_level:
        bonustring = '-ponly_first_level'
    pickled_string=img+bonustring+'-%s-%s-%dlevels'%(path_type,wavelet,levels)
    
    i = rbepwt.Image()
    i.read(filepath)
    if path_type != 'tensor':
        print("Working on segementation of image %s ..." % filepath)
        t0 = time.time()
        i.segment(scale=200,sigma=2,min_size=10)
        t1 = time.time()
        print('Segmenting took %s seconds' % (t1 - t0))
    #i.segment(method='kmeans',nclusters=30)
    print("Encoding image %s ..." % filepath)
    if path_type == 'tensor':
        t0 = time.time()
        i.encode_dwt(levels,wavelet)
        t1 = time.time()
    else:
        t0 = time.time()
        i.encode_rbepwt(levels,wavelet,path_type=path_type,paths_first_level = paths_only_at_first_level)
        t1 = time.time()
    print('Encoding took %s seconds' % (t1 - t0))
    if save:
        print("Saving encoded pickle as: %s" % pickleddir + pickled_string)
        i.save_pickle(pickleddir + pickled_string)
    return(pickled_string)

def threshold_decode(imgpath,thresh,filepath,path_type='easypath',save=True):
    img = rbepwt.Image()
    img.load_pickle(imgpath)
    img.threshold_coefs(thresh)
    if path_type == 'tensor':
        t0 = time.time()
        img.decode_rbepwt()
        t1 = time.time()
    else:
        t0 = time.time()
        img.decode_dwt()
        t1 = time.time()
    print('Decoding took %s seconds' % (t1 - t0))
    if save:
        print("Saving decoded pickle (%d coeffs) as: %s" % (thresh,filepath))
        img.save_pickle(filepath)
    
def for_error_plot():
    thresholds = [4096,2048,1024,512]
    img_names = ['cameraman256','house256','peppers256']
    #encodings = ['easypath','gradpath','epwt-easypath','tensor']
    #encodings = ['gradpath','epwt-easypath','tensor']
    encodings = ['tensor']
    for enc in encodings:
        for img in img_names:
            fpath = '../img/' + img +'.png'
            if enc == 'tensor':
                lev = 4
            else:
                lev = 16
            encoded_img_pickle_string = encode(fpath,path_type=enc,levels=lev)
            for thresh in thresholds:
                threshold_decode(pickleddir + encoded_img_pickle_string,thresh,decodedpickleddir+encoded_img_pickle_string+'--'+str(thresh))

if __name__ == '__main__':
    #encode(sys.argv[1])
    for_error_plot()
