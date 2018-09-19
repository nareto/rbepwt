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


#This script compares the "fast" decode procedure (rbepwt.decode()), which uses the region collections
#obtained during encoding, with the "full" one, which actually recomputes the paths from scratch as described in the paper

import rbepwt
import matplotlib.pyplot as plt
import numpy as np
import timeit

show_decodes = True
threshold = True
full_decode = True
wav = 'bior4.4'
#wav = 'haar'
levels = 12
#img = 'gradient64'
#img = 'sampleimg4'
img = 'house256'
#img = 'cameraman256'
#ext = '.jpg'
ext = '.png'
ptype = 'easypath'
#ptype = 'gradpath'
imgpath = 'img/'+img+ext
pickledpath='../pickled/'+img+'-%s-%s-%dlevels'%(ptype,wav,levels)
#pickledpath = '../pickled/gradient64-easypath-haar-12levels'
ncoefs = 51

fasti = rbepwt.Image()
#fasti.load_or_compute(imgpath,pickled_string,levels,wav)
fasti.load_pickle(pickledpath)
if threshold:
    fasti.rbepwt.threshold_coefs(ncoefs)
start = timeit.default_timer()
fasti.decode_rbepwt()
tot_time = timeit.default_timer() - start
print("psnr of fast decode: %f " %fasti.psnr())
print("tot time of decode:", tot_time)
if show_decodes:
    fasti.show_decoded(title = 'Fast Decode')

if full_decode:
    fulli = rbepwt.Image()
    fulli.load_pickle(pickledpath)
    if threshold:
        fulli.rbepwt.threshold_coefs(ncoefs)
    start = timeit.default_timer()
    fdi = rbepwt.full_decode(fulli.rbepwt.wavelet_details,fulli.rbepwt.region_collection_at_level[levels+1].values,fulli.label_img,wav,ptype)
    tot_time = timeit.default_timer() - start
    print("psnr of full decode: %f " % rbepwt.psnr(fulli.img,fdi))
    print("tot time of decode:", tot_time)
    p = rbepwt.Picture()
    p.load_array(fdi)
    if show_decodes:
        p.show(title = 'Full Decode')

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
