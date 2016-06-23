import ipdb
import rbepwt
import matplotlib.pyplot as plt
import numpy as np

wav = 'bior4.4'
#wav = 'haar'
levels = 12
#levels = 8
#levels = 2
#img = 'gradient64'
#img = 'sampleimg4'
img = 'house256'
#ext = '.jpg'
ext = '.png'
#pickled_string='house256-%dlevels'%levels
pickled_string=img+'-%s-%dlevels'%(wav,levels)
#pickled_string='sampleimg4-%dlevels'%levels
coefs_perc = 0.1
ncoefs = 6553

img = rbepwt.Image()
try:
    img.load('pickled/'+pickled_string)
except FileNotFoundError:
    img.read('img/'+img+ext)
    img.segment(scale=200,sigma=0.8,min_size=10)
    img.encode_rbepwt(levels,wav)
    img.save('pickled/'+pickled_string)
img.rbepwt.threshold_by_percentage(coefs_perc)
#ipdb.set_trace()
img.decode_rbepwt()

print("PERCENTAGE THRESHOLDED: %f" % coefs_perc)
print("Nonzero coefs: %d \npsnr of fast decode: %f " %(img.nonzero_coefs(),img.psnr()))
#print("wdetails: %s" % img.rbepwt.wavelet_details)
#print("wapprox: %s" % img.rbepwt.region_collection_dict[levels+1].points)
img.show_decoded(title='Percentage thresholded')

img.load('pickled/'+pickled_string)
img.rbepwt.threshold_coefs(ncoefs)
#ipdb.set_trace()
img.decode_rbepwt()

print("THRESHOLDED: %d" % ncoefs)
print("Nonzero coefs: %d \npsnr of fast decode: %f " %(img.nonzero_coefs(),img.psnr()))
#print("wdetails: %s" % img.rbepwt.wavelet_details)
#print("wapprox: %s" % img.rbepwt.region_collection_dict[levels+1].points)
img.show_decoded(title='Normal thresholded')
