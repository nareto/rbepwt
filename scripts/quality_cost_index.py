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

import matplotlib.pyplot as plt
import numpy as np
import rbepwt

sparsity = [4096,2048,1024,512]
image_names = ['cameraman256']
#encodings = ['easypath','gradpath','epwt-easypath','tensor']
#encodings = ['easypath','tensor']
#encodings = ['tbes0.0005']
#encodings = ['tensor']
#encodings = ['epwt-easypath']
encodings = ['easypath']

imgpath = '../img/'
savedir = '../decoded_pickles-euclidean/'

def plot(image_name,bits=8):
    #encodings = ['gradpath','tensor']
    for e in encodings:
        qri = np.zeros(len(sparsity))
        if e == 'easypath' or e[:4] == 'tbes':
            #lsty = '-'
            lsty = 'None'
            mrkr = '_'
            lco = 'g'
        elif e == 'gradpath':
            #lsty = '-.'
            lsty = 'None'
            mrkr = '|'
            lco = 'r'
        elif e == 'epwt-easypath':
            #lsty = '--'
            lsty = 'None'
            mrkr = 'x'
            lco = 'b'
        elif e == 'tensor':
            #lsty = ':'
            lsty = 'None'
            lco = 'k'
            mrkr = '.'
        for idx,s in enumerate(sparsity):
            #print("working on %s with encoding %s and threshold %d" % (imgname,enc,thresh))
            img = rbepwt.Image()
            if e == 'tensor':
                levs = '4'
                #levs = '8'
            else:
                levs = '16'
            loadstr = savedir+image_name+'-'+e+'-bior4.4'+'-'+levs+'levels--'+str(s)
            #loadstr = savedir+imgname+'-'+enc+'-haar'+'-'+levs+'levels--'+str(thresh)
            #print('Loading pickle: %s ' % loadstr)
            img.load_pickle(loadstr)
            img.segmentation_method = 'None(Tensor)'
            segmcost = 0
            if e != 'tensor':
                img.segmentation.__build_borders_set__()
                img.segmentation.compute_encoding()
                img.segmentation_method = 'Felzenszwalb-Huttenlocher'
                #img.segmentation_method = 'TBES'
                segmcost = img.segmentation.compute_encoding_length()
            sparse_coding_cost = img.sparse_coding_cost(bits)
            cost = img.encoding_cost(bits)
            val = img.quality_cost_index(bits)
            qri[idx] = val
            plt.plot(sparsity,qri,color=lco,linestyle=lsty,marker=mrkr)#,markersize=msize,markeredgewidth=2)
            orgmodestr = '|'+image_name+'|'+img.segmentation_method+'|'+str(img.nonzero_coefs())+'|'+str(bits)+'|'+str(segmcost)+'|'+str(sparse_coding_cost)+'|'+str(cost)+'|'+str(val)+'|'+str(img.psnr())+'|'+str(img.haarpsi())+'|'
            print(orgmodestr)
    plt.xticks(sparsity)
    plt.xlim(420,5000)
    plt.legend()
    plt.show()
