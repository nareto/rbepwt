import rbepwt
import matplotlib.pyplot as plt

levels = 3
i = rbepwt.Image()
#i.read('img/sampleimg4x4.png')
i.read('img/gradient.jpg')
i.segment(scale=200,sigma=0.8,min_size=10)
i.encode_rbepwt(levels,'db1')
#i.save('pickled/gradient-3lev')
#i.load('pickled/gradient-3lev')

decoded_img = rbepwt.full_decode(i.rbepwt.wavelet_details,i.rbepwt.region_collection_dict[levels+1],i.segmentation.label_img,'db1')

fig = plt.figure()
plt.imshow(decoded_img,cmap=plt.cm.gray,interpolation='nearest')
plt.show()