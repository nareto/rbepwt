import ipdb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import pickle
import rbepwt

thresholds = [512]
#images = ['peppers']
images = ['peppers','cameraman','house']
encs = ['epwt-easypath-bior4.4-16levels-euclidean','tensor-bior4.4-4levels','easypath-bior4.4-16levels-euclidean','gradpath-bior4.4-16levels-euclidean']
#encs = ['epwt-easypath-bior4.4-16levels-euclidean','tensor-bior4.4-4levels','gradpath-bior4.4-16levels-euclidean']
#encs = ['tensor-bior4.4-4levels']
imgpath = '../img/'
savedir = '../decoded_pickles-euclidean/'
export_dir = '/Users/renato/ownCloud/phd/talks-papers/rbepwt-canazeiproceedings/img/'

def encode_threshold_and_save(scale=200,sigma=2,min_size=10):
    for enc in encs:
        if enc[:6] == 'tensor':
            method = 'tensor'
        else:
            method = enc.rstrip('bior4.4-16levels-euclidean')
        for img in images:
            rbimg = rbepwt.Image()
            loadpath = imgpath+img+'256.png'
            rbimg.read(loadpath)
            if method in ['easypath','gradpath']:
                rbimg.segment(scale=scale,sigma=sigma,min_size=min_size)
                rbimg.encode_rbepwt(16,'bior4.4',path_type=method,euclidean_distance=True)
            elif method == 'epwt-easypath':
                rbimg.encode_epwt(16,'bior4.4')
            elif method == 'tensor':
                rbimg.encode_dwt(4,'bior4.4')
            idfullstring = img+'256-'+enc+'--full'
            savepath = savedir+idfullstring
            print("Saving in %s" % savepath)
            rbimg.save_pickle(savepath)
            for thresh in thresholds:
                rbimg = rbepwt.Image()
                loadpath = savedir+idfullstring
                #ipdb.set_trace()
                rbimg.load_pickle(loadpath)
                rbimg.threshold_coefs(thresh)
                if method in ['easypath','gradpath']:
                    rbimg.decode_rbepwt()
                elif method == 'epwt-easypath':
                    rbimg.decode_epwt()
                elif method == 'tensor':
                    rbimg.decode_dwt()
                idstring = img+'256-'+enc+'--'+str(thresh)
                savepath = savedir+idstring
                print("Saving in %s" % savepath)
                rbimg.save_pickle(savepath)
                savedecpath = export_dir+idstring+'.png'
                print("Saving in %s" % savedecpath)
                rbimg.save_decoded(filepath=savedecpath,title=None)


def save_decodings():
    for enc in encs:
        for thresh in thresholds:
            for img in images:
                rbimg = rbepwt.Image()
                idstring = '256-'+enc+'--'+str(thresh)
                loadpath = savedir+img+idstring
                rbimg.load_pickle(loadpath)
                savepath = export_dir+img+idstring+'.png'
                print("Saving in %s" % savepath)
                rbimg.save_decoded(filepath=savepath,title=None)


def save_segmentations():
    for img in images:
        enc = encs[0]
        thresh = thresholds[0]
        rbimg = rbepwt.Image()
        idstring = '256-'+enc+'--full'
        loadpath = savedir+img+idstring
        rbimg.load_pickle(loadpath)
        savepath = export_dir+img+'256-segmentation'+'.png'
        print("Saving in %s" % savepath)
        #ipdb.set_trace()
        rbimg.save_segmentation(filepath=savepath,title=None)

save_segmentations()
#encode_threshold_and_save()
