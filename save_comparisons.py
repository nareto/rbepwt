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
imgpath = 'img/'
savedir = 'decoded_pickles-euclidean/'
export_dir = '/Users/renato/ownCloud/phd/talks-papers/rbepwt-canazeiproceedings/img/'

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
