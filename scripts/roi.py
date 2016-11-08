import ipdb
import numpy as np
import rbepwt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import random

def find_intersecting_regions(img,rect):
    labels = set()
    npoints = 0
    for i in range(rect[0],rect[2]+1):
        for j in range(rect[1],rect[3]+1):
            labels.add(int(img.label_img[i,j]))
            npoints += 1
    print("%d points in the regions intersecting the rectangle" % npoints)
    return(labels)

def show_rectangle(img,rect):
    fig = plt.figure()
    plt.imshow(img.img,cmap=plt.cm.gray)
    ax = fig.gca()
    x = rect[1]
    y = rect[0]
    width = rect[3] - rect[1]
    height = rect[2] - rect[0]
    ax.add_patch(patches.Rectangle( (x, y),width,height,color = 'red',fill=False))
    plt.show()

def compute_roi_coeffs(img,regionsidx):
    #Save the coefficients we want by visiting the tree starting from the leafs in the selected regions
    coeffset = set()
    skipped_prev, prev_had_odd_len, prev_region_len = False, False, 0
    for level in range(1,img.rbepwt.levels+1):
        for regionidx, region in img.rbepwt.region_collection_at_level[level]:
            if skipped_prev + prev_had_odd_len == True:
                skip_first = True
            else:
                skip_first = False
                skipped_prev = skip_first
                prev_had_odd_len = len(region) % 2
            if regionidx in regionsidx:
                underregion = img.rbepwt.region_collection_at_level[level+1][regionidx]
                invperm = sorted(range(len(underregion)), key = lambda k: underregion.permutation[k])
                if not skip_first:
                    bpoints = enumerate(region.base_points)
                else:
                    bpoints = enumerate(region.base_points[1:])
                for i,coord in bpoints:
                    newi = int(i/2)
                    #perm_newi = invperm[newi]
                    perm_newi = underregion.permutation[newi]
                    #perm_newi = random.choice(invperm)
                    coeffset.add((regionidx,level,perm_newi))

    #Keep the coefficients in coeffset and set to 0 all the others
    for level in range(1, img.rbepwt.levels+1):
        prev_reg_len = 0
        wd = img.rbepwt.wavelet_details[level]
        for regionidx,region in img.rbepwt.region_collection_at_level[level+1]:
            if regionidx not in regionsidx:
                img.rbepwt.wavelet_details[level][prev_reg_len:prev_reg_len + len(region)] = np.zeros(len(region))
                prev_reg_len += len(region)
                continue
            for idx,value in enumerate(wd[prev_reg_len:prev_reg_len + len(region)]):
                #print("idx = %s. Checking: %s,%s,%s -- prev_reg_len = %d" %  (idx, regionidx,level,idx,prev_reg_len) )
                #if idx < prev_reg_len or idx >= prev_reg_len + len(wd) or (regionidx,level,idx) not in coeffset:
                if (regionidx,level,idx) not in coeffset:
                    #img.rbepwt.wavelet_details[level][prev_reg_len + idx] = 0
                    img.rbepwt.wavelet_details[level][prev_reg_len+idx] = 0
                    #print("set to 0")
                    if level == img.rbepwt.levels:
                        img.rbepwt.region_collection_at_level[level+1].values[prev_reg_len+idx] = 0
            prev_reg_len += len(region)
    print("img.nonzero_coefs() = %d" % img.nonzero_coefs())

