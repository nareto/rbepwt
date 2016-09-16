import ipdb
import rbepwt
import matplotlib.pyplot as plt
import numpy as np

wav = 'bior4.4'
#wav = 'haar'
#levels = 12
levels = 16
#levels = 2
#img = 'gradient64'
#img = 'sampleimg4'
#img = 'house256'
img = 'cameraman256'
#ext = '.jpg'
ext = '.png'
ptype = 'easypath'
#ptype = 'gradpath'
#pickled_string='house256-%dlevels'%levels
imgpath = 'img/'+img+ext
ncoefs = 512

roi_img = rbepwt.Image()
roi_img.read(imgpath)
roi_img.segment(scale=250,sigma=2,min_size=10)
roi_img.label_img[:,0:128] = roi_img.label_img.max()
roi_img.segmentation.compute_label_dict()
roi_img.has_segmentation = True
roi_img.segmentation.label_img = roi_img.label_img

roi_img.encode_rbepwt(levels, wav, ptype)
roi_img.rbepwt.threshold_coefs(ncoefs)
#ipdb.set_trace()
roi_img.decode_rbepwt()
roi_img.show_decoded()
