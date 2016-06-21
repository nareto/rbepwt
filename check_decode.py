import rbepwt
import matplotlib.pyplot as plt
import numpy as np
wav = 'bior4.4'
#wav = 'haar'
levels = 8
img = 'gradient64'
ext = '.jpg'
#pickled_string='house256-%dlevels'%levels
pickled_string=img+'-%s-%dlevels'%(wav,levels)
#pickled_string='sampleimg4-%dlevels'%levels
ncoefs = 100

fasti = rbepwt.Image()
try:
    fasti.load('pickled/'+pickled_string)
except FileNotFoundError:
    fasti.read('img/'+img+ext)
    fasti.segment(scale=200,sigma=0.8,min_size=10)
    fasti.encode_rbepwt(levels,wav)
    fasti.save('pickled/'+pickled_string)
fasti.rbepwt.threshold_coefs(ncoefs)
fasti.decode_rbepwt()
print("psnr of fast decode: %f " %fasti.psnr())
#fasti.show_decoded()

fulli = rbepwt.Image()
fulli.load('pickled/'+pickled_string)
fulli.rbepwt.threshold_coefs(ncoefs)
fdi = rbepwt.full_decode(fulli.rbepwt.wavelet_details,fulli.rbepwt.region_collection_dict[levels+1],fulli.label_img,wav)
print("psnr of full decode: %f " % rbepwt.psnr(fulli.img,fdi))
p = rbepwt.Picture()
p.load_array(fdi)
#p.show('Full decode')

print('Wavelet dict differences:')
rbepwt.compare_wavelet_dicts(fasti.rbepwt.wavelet_coefs_dict(),fulli.rbepwt.wavelet_coefs_dict())

fasti_coeffs = fasti.rbepwt.flat_wavelet()
fulli_coeffs = fulli.rbepwt.flat_wavelet()
#plt.subplot(2,1,1)
#plt.plot(fasti_coeffs)
#plt.subplot(2,1,2)
#plt.plot(fulli_coeffs)
#plt.plot(np.abs(fasti_coeffs-fulli_coeffs))
plt.imshow(np.abs(fasti.decoded_img - fdi),interpolation='nearest')
plt.colorbar()
plt.show()
