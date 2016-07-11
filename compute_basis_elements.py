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
#img_codenames = [('peppers256-easypath-bior4.4-16levels',(0,20),(1,3),2)]
#img_codenames = [('gradient64-easypath-bior4.4-12levels',(0,4),(13,0),2)]
img_codenames = [('peppers256-easypath-bior4.4-16levels',(0,20),(17,0),3),\
                 ('peppers256-gradpath-bior4.4-16levels',(0,20),(17,0),3),\
                 ('peppers256-epwt-easypath-bior4.4-16levels',(0,20),(17,0),3),\
                 ('house256-easypath-bior4.4-16levels',(0,20),(17,0),3),\
                 ('house256-gradpath-bior4.4-16levels',(0,20),(17,0),3),\
                 ('house256-epwt-easypath-bior4.4-16levels',(0,20),(17,0),3),\
                 ('cameraman256-easypath-bior4.4-16levels',(0,20),(17,0),3),\
                 ('cameraman256-gradpath-bior4.4-16levels',(0,20),(17,0),3),\
                 ('cameraman256-epwt-easypath-bior4.4-16levels',(0,20),(17,0),3),\
]
save = True

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
    #ipdb.set_trace()
    if approx:
        img.rbepwt.region_collection_at_level[curlevel].values[i] = 1
    else:
        img.rbepwt.wavelet_details[curlevel][curidx] = 1
    img.decode_rbepwt()
    return(img)

def compute_basis_at_level(img_codename,level,i):
    imgpath = 'pickled/' + img_codename

    img = rbepwt.Image()
    img.load_pickle(imgpath)
    #set all coefficients to 0
    curlevel = img.rbepwt.levels + 1
    prevlen = 0
    approxvec = img.rbepwt.region_collection_at_level[curlevel].values    
    curlen = len(approxvec)
    while curlevel > level:
        prevlen += curlen
        curlevel -= 1
        curlen = len(img.rbepwt.wavelet_details[curlevel])
    return(compute_basis(img_codename,prevlen+i))
        
def compute_basis_range(img_codename,basis_range):
    savedir = 'basis_elements/'

    low,high = basis_range
    for i in range(low,high):
        img = compute_basis(img_codename,i)
        if save:
            img.save_decoded(title=None,filepath=savedir+img_codename+'-basis'+str(i)+'.png')
        else:
            img.show_decoded(title='Basis element %d' % i) 

def find_n_largest_coef(img_codename,level,n):
    """Returns the index of the n largest coeffs at level"""
    
    imgpath = 'pickled/' + img_codename
    img = rbepwt.Image()
    img.load_pickle(imgpath)
    
    if level == img.rbepwt.levels + 1:
        maxlevels = img.rbepwt.levels
        vec = img.rbepwt.region_collection_at_level[maxlevels+1].values
    else:
        vec = img.rbepwt.wavelet_details[level]

    sortidx = np.argsort(vec)
    return(sortidx[:n])

#for ic,basis_range,levels_range,n in img_codenames:
#    compute_basis_range(ic,basis_range)

savedir='basis_elements-best/'
for ic,basis_range,levels_range,n in img_codenames:
    low,high = levels_range
    step = 1
    if low > high:
        step = -1
    for lev in range(low,high,step):
        for greatest,i in enumerate(find_n_largest_coef(ic,lev,n)):
            img = compute_basis_at_level(ic,lev,i)
            if save:
                fname = savedir+ic+'-level'+str(lev)+'-greatest'+str(greatest)+'-coef'+str(i)+'.png'
                img.save_decoded(title=None,filepath=fname)
            else:
                img.show_decoded(title='Level = %d, coef = %d' % (lev,i))
