import rbepwt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

img = rbepwt.Image()
#img.load_pickle('../decoded_pickles-euclidean/cameraman256-easypath-bior4.4-16levels--512')
img.load_pickle('../decoded_pickles-euclidean/peppers256-easypath-bior4.4-16levels--512')

#i,j,I,J where i and j are coords for top left point and I and J for bottom right
rect = (40,90, 110,150)

labels = set()
for i in range(rect[0],rect[2]+1):
    for j in range(rect[1],rect[3]+1):
        labels.add(int(img.label_img[i,j]))

#fig = plt.figure()
#plt.imshow(img.img,cmap=plt.cm.gray)
#ax = fig.gca()
#x = rect[1]
#y = rect[0]
#width = rect[3] - rect[1]
#height = rect[2] - rect[0]
#ax.add_patch(patches.Rectangle( (x, y),width,height,color = 'red',fill=False))
#plt.show()

        
coeffset = set()
for regionidx in labels:
    region = img.rbepwt.region_collection_at_level[1][int(regionidx)]
    for level in range(1,img.rbepwt.levels):
        underregion = img.rbepwt.region_collection_at_level[level+1][regionidx]
        invperm = sorted(range(len(underregion)), key = lambda k: underregion.permutation[k])
        for i,idx in enumerate(region.base_points):
            newi = int(i/2)
            perm_newi = invperm[newi]
            #perm_new = underregion.permutation[newi]
            coeffset.add((regionidx,level+1,perm_newi))

for level in range(1, img.rbepwt.levels):
    prev_reg_len = 0
    for regionidx,region in imb.rbepwt.region_collection_at_level[level]:
        if regionidx not in labels:
            prev_reg_len += len(region)
            continue
        wd = img.rbepwt.wavelet_details[level][prev_reg_len:prev_reg_len + len(region)]
        for idx,value in enumerate(wd):
            if (regionidx,level,idx) not in coeffset:
                img.rbepwt.wavelet_details[level][prev_reg_len + idx] = 0
        prev_reg_len += len(region)

img.decode_rbepwt()
img.show_decoded()
