import ipdb
import rbepwt
import numpy as np
import timeit

save = True
show_segmentation = False
levels = 8
wav = 'bior4.4'
#ptype = 'epwt-easypath'
ptype = 'easypath'
#ptype = 'gradpath'
img = 'gradient64'
#img = 'sampleimg4'
#img = 'house256'
#img = 'cameraman256'
#img = 'peppers256'
#ext = '.png'
ext = '.jpg'

imgpath = 'img/'+img+ext
pickled_string='pickled/'+img+'-%s-%s-%dlevels'%(ptype,wav,levels)

i = rbepwt.Image()
#i.read('img/cameraman.png')
i.read(imgpath)
#i.read('img/sampleimg4x4.png')
#i.segment(scale=2,sigma=0,min_size=1)
if ptype != 'epwt-easypath':
    print("Segmenting image...")
    i.segment(scale=200,sigma=2,min_size=10)
    if show_segmentation:
        i.segmentation.show()
print("Encoding image...")
start_time = timeit.default_timer()
i.encode_rbepwt(levels,wav,path_type=ptype)
time = timeit.default_timer() - start_time
print("Encoding required %f seconds" % time)
if save:
    i.save_pickle(pickled_string)
