import ipdb
import rbepwt
import numpy as np

save = True
show_segmentation = False
levels = 8
wav = 'bior4.4'
#ptype = 'easypath'
ptype = 'gradpath'
img = 'gradient64'
#img = 'sampleimg4'
#img = 'house256'
#img = 'peppers256'
#ext = '.png'
ext = '.jpg'
#ext = '.png'
#ptype = 'easypath'
ptype = 'gradpath'
imgpath = 'img/'+img+ext
pickled_string='pickled/'+img+'-%s-%s-%dlevels'%(ptype,wav,levels)

i = rbepwt.Image()
#i.read('img/cameraman.png')
i.read(imgpath)
#i.read('img/sampleimg4x4.png')
#i.segment(scale=2,sigma=0,min_size=1)
i.segment(scale=200,sigma=2,min_size=10)
if show_segmentation:
    i.segmentation.show()
i.encode_rbepwt(levels,wav,path_type=ptype)
if save:
    i.save(pickled_string)
#i.decode_rbepwt()
#i.show()
#i.show_decoded()
#i.rbepwt.show_wavelets()
#for key,wav in i.rbepwt.wavelet_details.items():
#    print("Wavelet coefs at level %d: %s" % (key,wav))
#print("Wavelet approx coefs: %s" % i.rbepwt.region_collection_at_level[levels + 1].values)
#i.rbepwt.threshold_coefs(4)
#for key,wav in i.rbepwt.wavelet_details.items():
#    print("Wavelet coefs at level %d: %s" % (key,wav))
#print("Wavelet approx coefs: %s" % i.rbepwt.region_collection_at_level[levels + 1].values)
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
