import sys,os
import rbepwt
import numpy as np

pickleddir = '../pickled/'
decodedpickleddir = '../decoded_pickles-euclidean/'

def encode(filepath,wavelet='bior4.4',levels=16,path_type='easypath',paths_only_at_first_level=False,save=True):
    filename,ext = os.path.splitext(filepath)
    img = filename.split('/')[-1]
    bonustring = ''
    if paths_only_at_first_level:
        bonustring = '-ponly_first_level'
    pickled_string=img+bonustring+'-%s-%s-%dlevels'%(path_type,wavelet,levels)
    print("Will save encoded pickle as: %s" % pickleddir + pickled_string)
    
    i = rbepwt.Image()
    i.read(filepath)
    print("Working on segementation of image %s ..." % filepath)
    i.segment(scale=200,sigma=2,min_size=10)
    #i.segment(method='kmeans',nclusters=30)
    print("Encoding image %s ..." % filepath)
    i.encode_rbepwt(levels,wavelet,path_type=path_type,paths_first_level = paths_only_at_first_level)
    if save:
        i.save_pickle(pickleddir + pickled_string)
    return(pickled_string)

def threshold_decode(imgpath,thresh,filepath,save=True):
    img = rbepwt.Image()
    img.load_pickle(imgpath)
    img.threshold_coeffs(thresh)
    img.decode_rbepwt()
    if save:
        imd.save_pickle(filepath)
    
def for_error_plot():
    thresholds = [4096,2048,1024,512]
    img_names = ['cameraman256','house256','peppers256']
    for thresh in thresholds:
        for img in img_names:
            fpath = '../img/' + img +'.png'
            encoded_img_pickle_string = encode(fpath)
            threshold_decode(pickleddir + encoded_img_pickle_string,thresh,decodedpickleddir+pickleddir+'--'+str(thresh))

if __name__ == '__main__':
    #encode(sys.argv[1])
    for_error_plot()
