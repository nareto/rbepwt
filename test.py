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
i.rbepwt.show()
#p = i.rbepwt.paths[1][0]
#p = p.lazy_path()
#p.show()
#print(p.top_left,p.bottom_right)
#print('\n',p.base_points)
#i.segmentation.compute_label_dict()
#i.segmentation.compute_nlabels()
#print(i.segmentation.nlabels)
#print(len(i.segmentation.label_dict.keys()))
#i.show_segmentation()



#p = i.rbepwt.paths[1][5]
#p2= i.rbepwt.paths[1][6]
#c = p + p2
#print(c.points,'\n')
#for i in range(5):
#    #c = c.reduce_points(True)
#    c = c.reduce_points(False)
#    print(c.points,'\n')
#c.show()
#c = c.lazy_path()
#print("lazy pathed:\n",c.base_points)
#c.show()
#p.show()

