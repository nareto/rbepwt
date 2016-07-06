import ipdb
import matplotlib.pyplot as plt
#import pylab as plt
import numpy as np
import skimage.io
plt.rcParams['figure.figsize'] = (12, 12)
import rbepwt

img_codenames = [('gradient64-easypath-bior4.4-12levels',20),\
                 ('house256-gradpath-bior4.4-16levels',20),\
                 ('peppers256-easypath-bior4.4-12levels',20)]


def compute_basis(img_codename,nbasis):
    imgpath = 'pickled/' + img_codename
    #nbasis = 20
    savedir = 'basis_elements/'

    img = rbepwt.Image()
    img.load_pickle(imgpath)
    prevlen = 0
    curlevel = img.rbepwt.levels + 1
    approxvec = img.rbepwt.region_collection_at_level[curlevel].values
    approx = True
    for i in range(nbasis):
        print("\n--COMPUTING BASIS ELEMENT %d FOR image %s--\n" % (i,img_codename))
        img = rbepwt.Image()
        img.load_pickle(imgpath)
        for lev in range(1,img.rbepwt.levels+1):
            img.rbepwt.wavelet_details[lev] = np.zeros_like(img.rbepwt.wavelet_details[lev])
        img.rbepwt.region_collection_at_level[curlevel].values = np.zeros_like(approxvec)
        if i == len(approxvec) and approx:
            approx = False
            prevlen = len(approxvec)
            curlevel -= 1
        curidx = i - prevlen
        if approx:
            img.rbepwt.region_collection_at_level[curlevel].values[i] = 1
        else:
            #print("second",curidx)
            #print("first",i,curlevel,prevlen)
            img.rbepwt.wavelet_details[curlevel][curidx] = 1
            if curidx + 1== len(img.rbepwt.wavelet_details[curlevel]):
                prevlen += len(img.rbepwt.wavelet_details[curlevel])
                curlevel -= 1
        img.decode_rbepwt()
        #img.show_decoded(title='Basis element %d' % i) 
        img.save_decoded(title=None,filepath=savedir+img_codename+'-basis'+str(i)+'.png')   


for ic,nbasis in img_codenames:
    compute_basis(ic,nbasis)
