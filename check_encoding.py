import ipdb
import rbepwt
import numpy as np
import timeit


def encode_and_save(imgpath,ext,ptype,levels,wav,save=False,filepath=None,show_segmentation=False):
    i = rbepwt.Image()
    i.read(imgpath)
    if ptype != 'epwt-easypath':
        print("Segmenting image...")
        i.segment(scale=200,sigma=2,min_size=10)
        if show_segmentation:
            i.segmentation.show()
    print("Encoding image...")
    start_time = timeit.default_timer()
    i.encode_rbepwt(levels,wav,path_type=ptype)
    time = timeit.default_timer() - start_time
    print("Encoding required %f seconds" % time)
    if save:
        i.save_pickle(filepath)


def encode_many():
    levels = 16
    wav = 'bior4.4'
    ext = '.png'
    #images = ['peppers256','cameraman256','house256']
    #encoding_types = ['easypath','gradpath']
    images = ['cameraman256']
    encoding_types = ['gradpath']
    for img in images:
        for ptype in encoding_types:
            imgpath = 'img/'+img+ext
            #pickled_string='pickled/'+img+'-%s-%s-%dlevels-maxdist'%(ptype,wav,levels)
            #encode_and_save(imgpath,ext,ptype,levels,wav,save=True,filepath=pickled_string,show_segmentation=False)
            pickled_string='pickled/'+img+'-%s-%s-%dlevels-euclidean'%(ptype,wav,levels)
            encode_and_save(imgpath,ext,ptype,levels,wav,save=True,filepath=pickled_string,show_segmentation=False)
            
