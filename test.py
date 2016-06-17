import ipdb
import rbepwt
import numpy as np
levels = 4
wav = 'bior4.4'

i = rbepwt.Image()
#i.read('img/cameraman.png')
#i.read('img/gradient64.jpg')
i.read('img/sampleimg4x4.png')
i.segment(scale=2,sigma=0,min_size=1)
#i.segment(scale=200,sigma=0.8,min_size=10)
i.encode_rbepwt(levels,wav)
#ipdb.set_trace()
#i.rbepwt.show()
#i.rbepwt.show_wavelets()
#i.rbepwt.threshold_coeffs(1)
i.decode_rbepwt()
#i.show()
i.show_decoded()
#i.rbepwt.show_wavelets()
#for key,wav in i.rbepwt.wavelet_details.items():
#    print("Wavelet coefs at level %d: %s" % (key,wav))
#print("Wavelet approx coefs: %s" % i.rbepwt.region_collection_dict[levels + 1].values)
#i.rbepwt.threshold_coefs(4)
#for key,wav in i.rbepwt.wavelet_details.items():
#    print("Wavelet coefs at level %d: %s" % (key,wav))
#print("Wavelet approx coefs: %s" % i.rbepwt.region_collection_dict[levels + 1].values)
#i.decode_rbepwt()
##i.show_decoded()
#i.rbepwt.show_wavelets()
#i.rbepwt.show_wavelets()
#i.show_segmentation()


#R1 = rbepwt.Region([(1,2),(1,5)],[2,3])
#r2 = rbepwt.Region([(1,2),(1,50),(34,23),(92,2)],[2,3,4,5])
#r3 = r1 + r2
#print(r3.points, r3.subregions)
#r4 = r1.merge(r2,copy_regions=True)
#print(r4.points, r4.subregions)
#r1.points=[(213,234),(234,234234)]
#print(r4.subregions[0].points, r4.subregions[1].points)
#print(r1.points)
