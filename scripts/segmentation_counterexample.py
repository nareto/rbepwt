import numpy as np
import matplotlib.pyplot as plt
import rbepwt

n = 50
length = 0.7
z1 = np.ones((n,n/2))
z2 = np.ones((n,n/2))
for (i,j),v in np.ndenumerate(z2):
    z2[i,j] = min(1,i/(n*length))
z = np.concatenate((z1,z2),axis=1)
print("greyvalue increment: %f" % (1/(n*length)))
img = rbepwt.Image()
img.read_array(z)
img.segment(scale=n*n*5,sigma=0,min_size=10)
img.show_segmentation(colorbar=False,border=True)
#img.show(title=None,border=True)

#plt.imshow(z,cmap=plt.cm.gray,interpolation='none')
#plt.show()
