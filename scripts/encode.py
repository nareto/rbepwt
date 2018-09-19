#    Copyright 2017 Renato Budinich
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#


import sys,os
import rbepwt
import numpy as np
import time

pickleddir = '../pickled-tbes/'
decodedpickleddir = '../decoded_pickles-tbes/'

def encode(filepath,wavelet='bior4.4',levels=16,path_type='easypath',loadsegm=None,paths_only_at_first_level=False,save=True):
    filename,ext = os.path.splitext(filepath)
    img = filename.split('/')[-1]
    
    i = rbepwt.Image()
    i.read(filepath)
    if path_type != 'tensor':
        if loadsegm is None:
            print("Working on segementation of image %s ..." % filepath)
            t0 = time.time()
            i.segment(scale=200,sigma=2,min_size=10)
            t1 = time.time()
            print('Segmenting took %s seconds' % (t1 - t0))
        else:
            i.load_mat_segmentation(loadsegm)
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
    bonustring = ''
    if paths_only_at_first_level:
        bonustring = '-ponly_first_level'
    if i.segmentation_method == 'tbes':
        bonustring +='-tbes'
    pickled_string=img+bonustring+'-%s-%s-%dlevels'%(path_type,wavelet,levels)
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
        img.decode_dwt()
        t1 = time.time()
    else:
        t0 = time.time()
        img.decode_rbepwt()
        t1 = time.time()
    print('Decoding took %s seconds' % (t1 - t0))
    if img.segmentation_method == 'tbes':
        filepath += '--tbes'
    if save:
        print("Saving decoded pickle (%d coeffs) as: %s" % (thresh,filepath))
        img.save_pickle(filepath)
    
def for_error_plot():
    thresholds = [4096,2048,1024,512]
    img_names = ['cameraman256']
    #img_names = ['cameraman256','house256','peppers256']
    encodings = ['easypath','gradpath','epwt-easypath','tensor']
    #encodings = ['gradpath','epwt-easypath','tensor']
    #encodings = ['tensor']
    precomputed_segmentations = ['../tbes_1.0/cameraman0.0005.mat']
    for enc in encodings:
        for img,psegm in zip(img_names,precomputed_segmentations):
            fpath = '../img/' + img +'.png'
            if enc == 'tensor':
                lev = 4
            else:
                lev = 16
            encoded_img_pickle_string = encode(fpath,path_type=enc,levels=lev,loadsegm=psegm)
            for thresh in thresholds:
                threshold_decode(pickleddir + encoded_img_pickle_string,thresh,decodedpickleddir+encoded_img_pickle_string+'--'+str(thresh),path_type=enc)

if __name__ == '__main__':
    #encode(sys.argv[1])
    for_error_plot()
