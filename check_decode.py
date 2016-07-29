import rbepwt
import matplotlib.pyplot as plt
import numpy as np
import timeit

show_decodes = True
threshold = False
full_decode = False
wav = 'bior4.4'
#wav = 'haar'
levels = 8
img = 'gradient64'
#img = 'sampleimg4'
#img = 'house256'
ext = '.jpg'
#ext = '.png'
ptype = 'easypath'
#ptype = 'gradpath'
imgpath = 'img/'+img+ext
pickled_string='pickled/'+img+'-%s-%s-%dlevels'%(ptype,wav,levels)
ncoefs = 500

fasti = rbepwt.Image()
#fasti.load_or_compute(imgpath,pickled_string,levels,wav)
fasti.load_pickle(pickled_string)
print(pickled_string)
if threshold:
    fasti.rbepwt.threshold_coefs(ncoefs)
start = timeit.default_timer()
fasti.decode_rbepwt()
tot_time = timeit.default_timer() - start
print("psnr of fast decode: %f " %fasti.psnr())
print("tot time of decode:", tot_time)
if show_decodes:
    fasti.show_decoded('Fast Decode')

if full_decode:
    fulli = rbepwt.Image()
    fulli.load_pickle(pickled_string)
    if threshold:
        fulli.rbepwt.threshold_coefs(ncoefs)
    start = timeit.default_timer()
    fdi = rbepwt.full_decode(fulli.rbepwt.wavelet_details,fulli.rbepwt.region_collection_at_level[levels+1],fulli.label_img,wav,ptype)
    tot_time = timeit.default_timer() - start
    print("psnr of full decode: %f " % rbepwt.psnr(fulli.img,fdi))
    print("tot time of decode:", tot_time)
    p = rbepwt.Picture()
    p.load_array(fdi)
    if show_decodes:
        p.show('Full Decode')

    print('Wavelet dict differences (should be empty list):')
    rbepwt.compare_wavelet_dicts(fasti.rbepwt.wavelet_coefs_dict(),fulli.rbepwt.wavelet_coefs_dict())

#fasti_coeffs = fasti.rbepwt.flat_wavelet()
#fulli_coeffs = fulli.rbepwt.flat_wavelet()
#plt.subplot(2,1,1)
#plt.plot(fasti_coeffs)
#plt.subplot(2,1,2)
#plt.plot(fulli_coeffs)
#plt.plot(np.abs(fasti_coeffs-fulli_coeffs))
#plt.imshow(np.abs(fasti.decoded_img - fdi),interpolation='nearest')
#plt.colorbar()
#plt.show()
