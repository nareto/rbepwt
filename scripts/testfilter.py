import matplotlib.pyplot as plt
import rbepwt
import numpy as np
import copy

cam = rbepwt.Image()
cam.load_pickle('../decoded_pickles-euclidean/cameraman256-easypath-bior4.4-16levels--512')
r = cam.rbepwt.region_collection_at_level[1][1]
r.filter(0.5)
rimg = r.get_enclosing_img()
plt.imshow(rimg,cmap=plt.cm.gray)
plt.show()
