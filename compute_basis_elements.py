import ipdb
import matplotlib.pyplot as plt
#import pylab as plt
import numpy as np
import skimage.io
#plt.rcParams['figure.figsize'] = (12, 12)
import rbepwt

#img_codenames = [('gradient64-easypath-bior4.4-12levels',20),\
#                 ('house256-gradpath-bior4.4-16levels',20),\
#                 ('peppers256-easypath-bior4.4-12levels',20)]
#img_codenames = [('peppers256-easypath-bior4.4-12levels',(17,20))]
#img_codenames = [('house256-easypath-bior4.4-16levels',(0,20))]
img_codenames = [('cameraman256-easypath-bior4.4-16levels',(0,20))]

def compute_basis(img_codename,i):
    print("\n--COMPUTING BASIS ELEMENT %d FOR image %s--\n" % (i,img_codename))

    imgpath = 'pickled/' + img_codename

    img = rbepwt.Image()
    img.load_pickle(imgpath)
    #set all coefficients to 0
    curlevel = img.rbepwt.levels + 1
    approxvec = img.rbepwt.region_collection_at_level[curlevel].values
    for lev in range(1,img.rbepwt.levels+1):
        img.rbepwt.wavelet_details[lev] = np.zeros_like(img.rbepwt.wavelet_details[lev])
    img.rbepwt.region_collection_at_level[curlevel].values = np.zeros_like(approxvec)
    #search where the i-th basis element is
    prevlen = 0
    curlen = len(approxvec)
    curidx = i
    if i >= curlen:
        approx = False
        while curidx >= curlen:
            curlevel -= 1
            prevlen += curlen
            curidx = i - prevlen
            curlen = len(img.rbepwt.wavelet_details[curlevel])
    else:
        approx = True
    #set the appropriate basis element to 1
    if approx:
        img.rbepwt.region_collection_at_level[curlevel].values[i] = 1
    else:
        img.rbepwt.wavelet_details[curlevel][curidx] = 1
    img.decode_rbepwt()
    return(img)


def compute_basis_range(img_codename,basis_range):
    savedir = 'basis_elements/'

    low,high = basis_range
    for i in range(low,high):
        img = compute_basis(img_codename,i)
        #img.show_decoded(title='Basis element %d' % i) 
        img.save_decoded(title=None,filepath=savedir+img_codename+'-basis'+str(i)+'.png')   


for ic,basis_range in img_codenames:
    compute_basis_range(ic,basis_range)
