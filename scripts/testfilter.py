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

#Here we tested the effects of convoluting the decoded values with a gaussian kernel.
#The results weren't promising and this wasn't included in the paper.

import matplotlib.pyplot as plt
import rbepwt
import numpy as np
import copy

cam = rbepwt.Image()
cam.load_pickle('../decoded_pickles-euclidean/cameraman256-easypath-bior4.4-16levels--512')
sigma = 1
#r = cam.rbepwt.region_collection_at_level[1][1]
#r.filter(0.5)
#rimg = r.get_enclosing_img()
#plt.imshow(rimg,cmap=plt.cm.gray)
#plt.show()
cam.filter(sigma)
haarpsi = cam.haarpsi()
filteredhaarpsi = cam.haarpsi(filtered=True)
print('%40s%10f\n%40s%10f' % ('HaarPSI of decoded image: ',haarpsi,'HaarPSI of filtered decoded image: ',filteredhaarpsi))
cam.show_decoded(title='Decoded')
cam.show_filtered(title='Filtered decoded')

