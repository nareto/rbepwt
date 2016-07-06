import ipdb
import rbepwt
import matplotlib.pyplot as plt
import numpy as np

wav = 'bior4.4'
#wav = 'haar'
#levels = 12
std_levels = 2
rbepwt_levels = 12
#levels = 2
img = 'gradient64'
#img = 'sampleimg4'
#img = 'house256'
ext = '.jpg'
#ext = '.png'
ptype = 'easypath'
#ptype = 'gradpath'
#pickled_string='house256-%dlevels'%levels
imgpath = 'img/'+img+ext
pickled_string='pickled/'+img+'-%s-%s-%dlevels'%(ptype,wav,levels)
#pickled_string='sampleimg4-%dlevels'%levels
ncoefs = 410

stdwav_img = rbepwt.Image()
stdwav_img.read(imgpath)
stdwav_img.encode_dwt(std_levels,wav)
stdwav_img.dwt.threshold_coefs(ncoefs)
stdwav_img.decode_dwt()
print("Nonzero coefs: %d " % stdwav_img.nonzero_dwt_coefs())
print("Error of standard wavelet transform")
stdwav_img.error()
stdwav_img.show_decoded(title='Standard wavelet transform')

rbepwt_img  = rbepwt.Image()
rbepwt_img.load_or_compute(imgpath,pickled_string,rbepwt_levels,wav)
rbepwt_img.rbepwt.threshold_coefs(ncoefs)
rbepwt_img.decode_rbepwt()
print("Nonzero coefs: %d " % rbepwt_img.nonzero_coefs())
print("Error of region based wavelet transform")
rbepwt_img.error()
rbepwt_img.show_decoded(title='Region based wavelet transform')
