import rbepwt
wav = 'bior4.4'
levels = 12
pickled_string='house256-%dlevels'%levels
ncoefs = 1024

i = rbepwt.Image()
i.load('pickled/'+pickled_string)
i.rbepwt.threshold_coefs(ncoefs)
i.decode_rbepwt()
print("psnr of fast decode: %f " %i.psnr())
i.show_decoded()

i = rbepwt.Image()
i.load('pickled/'+pickled_string)
i.rbepwt.threshold_coefs(ncoefs)
fdi = rbepwt.full_decode(i.rbepwt.wavelet_details,i.rbepwt.region_collection_dict[levels+1],i.label_img,wav)
print("psnr of full decode: %f " % rbepwt.psnr(i.img,fdi))
p = rbepwt.Picture()
p.load_array(fdi)
p.show('Full decode')
