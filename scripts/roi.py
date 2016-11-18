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
    selected_idx = []
    prev_len = 0
    for label, region in img.rbepwt.region_collection_at_level[1]:
        if label in regionsidx:
            for coord,value in region:
                selected_idx.append(prev_len + region.base_points.index(coord))
        prev_len += len(region)
    for level in range(1,img.rbepwt.levels+1):
        bpoints = []
        new_selected_idx = []
        global_perm = []
        prev_len = 0
        for label, region in img.rbepwt.region_collection_at_level[level]:
            bpoints += region.base_points
            underregion = img.rbepwt.region_collection_at_level[level+1][label]
            lenreg = len(underregion)
            back_label = -1
            while lenreg == 0:
                underregion = img.rbepwt.region_collection_at_level[level+1][label + back_label]
                lenreg = len(underregion)
                back_label -= 1
            for i in underregion.permutation:
                global_perm += [prev_len + i]
            prev_len += lenreg
        for idx,coord in enumerate(bpoints):
            if idx in selected_idx:
                halfidx = int(idx/2)
                coeffset.add((level,halfidx))
                new_selected_idx.append(global_perm.index(halfidx))
        selected_idx = new_selected_idx
    #print("coeffset = ", coeffset)
    for level in range(1, img.rbepwt.levels+1):
        for idx,val in enumerate(img.rbepwt.wavelet_details[level]):
            coeff = (level,idx)
            if coeff not in coeffset:
                img.rbepwt.wavelet_details[level][idx] = 0
                #print("setting to 0: ", level,idx,coeff)
            #else:
            #    print(coeff)
    print("img.nonzero_coefs() = %d" % img.nonzero_coefs())

