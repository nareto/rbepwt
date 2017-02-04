#    Copyright 2017 Renato Budinich
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

#This script was used to generate the "difficult" segmentation talked about in the paper

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
#img.segment(scale=n*n*5,sigma=0,min_size=10)
img.segment(scale=n*10,sigma=0,min_size=10)
img.show_segmentation(colorbar=False,border=True)
#img.show(title=None,border=True)

#plt.imshow(z,cmap=plt.cm.gray,interpolation='none')
#plt.show()
