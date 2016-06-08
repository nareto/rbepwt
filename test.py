import ipdb
import rbepwt
import numpy as np

i = rbepwt.Image()
#i.read('../cameraman.png')
i.read('../gradient64.jpg')
#i.read('../sampleimg4x4.png')
#i.segment(scale=2,sigma=0,min_size=1)
i.segment(scale=200,sigma=0.8,min_size=10)
i.encode_rbepwt(9,'db1')
#ipdb.set_trace()
i.rbepwt.show(range(1,7))
#i.rbepwt.show_wavelets()
#i.rbepwt.threshold_coeffs(1)
i.decode_rbepwt()
#i.show()
#i.show_decoded()
i.rbepwt.show_wavelets()
i.show_segmentation()
#r1 = rbepwt.Region([(1,2),(1,5)],[2,3])
#r2 = rbepwt.Region([(1,2),(1,50),(34,23),(92,2)],[2,3,4,5])
#r3 = r1 + r2
#print(r3.points, r3.subregions)
#r4 = r1.merge(r2,copy_regions=True)
#print(r4.points, r4.subregions)
#r1.points=[(213,234),(234,234234)]
#print(r4.subregions[0].points, r4.subregions[1].points)
#print(r1.points)
