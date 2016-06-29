import ipdb
import rbepwt
import matplotlib.pyplot as plt
import numpy as np

wav = 'bior4.4'
#wav = 'haar'
#levels = 12
levels = 8
#levels = 2
img = 'gradient64'
#img = 'sampleimg4'
#img = 'house256'
ext = '.jpg'
#ext = '.png'
#ptype = 'easypath'
ptype = 'gradpath'
#pickled_string='house256-%dlevels'%levels
imgpath = 'img/'+img+ext
pickled_string='pickled/'+img+'-%s-%s-%dlevels'%(ptype,wav,levels)
#pickled_string='sampleimg4-%dlevels'%levels
coefs_perc = 0.1
ncoefs = 410

perct_img = rbepwt.Image()
perct_img.load_or_compute(imgpath,pickled_string,levels,wav)
perct_img.rbepwt.threshold_by_percentage(coefs_perc)
#ipdb.set_trace()
perct_img.decode_rbepwt()

print("PERCENTAGE THRESHOLDED: %f" % coefs_perc)
print("Nonzero coefs: %d " % perct_img.nonzero_coefs())
perct_img.error()
#print("wdetails: %s" % perct_img.rbepwt.wavelet_details)
#print("wapprox: %s" % perct_img.rbepwt.region_collection_at_level[levels+1].points)
perct_img.show_decoded(title='Percentage thresholded')

normt_img  = rbepwt.Image()
normt_img.load(pickled_string)
normt_img.rbepwt.threshold_coefs(ncoefs)
#ipdb.set_trace()
normt_img.decode_rbepwt()

print("THRESHOLDED: %d" % ncoefs)
print("Nonzero coefs: %d" % normt_img.nonzero_coefs())
normt_img.error()
#print("wdetails: %s" % normt_img.rbepwt.wavelet_details)
#print("wapprox: %s" % normt_img.rbepwt.region_collection_at_level[levels+1].points)
normt_img.show_decoded(title='Normal thresholded')
