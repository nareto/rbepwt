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

noseg_img = rbepwt.Image()
noseg_img.read(imgpath)

#define a fake segmentation where the whole images is one unique region
noseg_img.segmentation = rbepwt.Segmentation(noseg_img.img)
noseg_img.label_img = np.zeros(noseg_img.img.shape)
noseg_img.segmentation.label_img = noseg_img.label_img
noseg_img.segmentation.compute_label_dict()
noseg_img.has_segmentation = True

#encode,threshold,decode,show
noseg_img.encode_rbepwt(levels, wav, ptype)
noseg_img.rbepwt.threshold_coefs(ncoefs)
noseg_img.decode_rbepwt()
noseg_img.show_decoded()
