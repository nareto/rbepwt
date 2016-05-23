import rbepwt

i = rbepwt.Image()
#i.read('../cameraman.png')
i.read('../gradient64.jpg')
i.info()
#i.show()

i.segment()
i.show_segmentation()
