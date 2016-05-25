import rbepwt
import numpy as np

i = rbepwt.Image()
#i.read('../cameraman.png')
i.read('../gradient64.jpg')
#i.read('../sampleimg4x4.png')
#i.info()
#i.show()

#i.segment()
i.compute_rbepwt()
#p = i.rbepwt.paths[1][0]
#p.show()
i.rbepwt.show()
#print(p.top_left,p.bottom_right)
#print('\n',p.base_points)
#i.segmentation.compute_label_dict()
#i.segmentation.compute_nlabels()
#print(i.segmentation.nlabels)
#print(len(i.segmentation.label_dict.keys()))
#i.show_segmentation()
