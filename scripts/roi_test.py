import rbepwt
import roi

img = rbepwt.Image()
img.read('../img/gradient64.jpg')
#img.read('../img/sampleimg4.png')
#img.read('../img/cameraman256.png')
levs = 2
#img.segment(scale=1,min_size=0,sigma=0)
img.segment(scale=200,min_size=10,sigma=2)
img.show_segmentation()
img.encode_rbepwt(levs,'haar')
labels = set([1])
#print("wd: %s \nwa: %s" % (img.rbepwt.wavelet_details,img.rbepwt.region_collection_at_level[levs+1].values))
roi.compute_roi_coeffs(img,labels)
#img.decode_rbepwt()
#img.show_decoded()
#print("wd: %s \nwa: %s" % (img.rbepwt.wavelet_details,img.rbepwt.region_collection_at_level[levs+1].values))
img.threshold_coefs(2)
#print("wd: %s \nwa: %s" % (img.rbepwt.wavelet_details,img.rbepwt.region_collection_at_level[levs+1].values))
fdi = rbepwt.full_decode(img.rbepwt.wavelet_details,img.rbepwt.region_collection_at_level[levs+1].values,img.label_img,'haar')
p = rbepwt.Picture()
p.load_array(fdi)
p.show()

#print("wd: %s \nwa: %s" % (img.rbepwt.wavelet_details,img.rbepwt.region_collection_at_level[levs+1].values))
