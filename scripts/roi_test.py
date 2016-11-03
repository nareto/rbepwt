import rbepwt
import roi

grad = rbepwt.Image()
grad.read('../img/gradient64.jpg')
grad.encode_rbepwt(8,'haar')
labels = set([1])
roi.compute_roi_coeffs(grad,labels)
#grad.decode_rbepwt()
#grad.show_decoded()
fdi = rbepwt.full_decode(grad.rbepwt.wavelet_details,grad.rbepwt.region_collection_at_level[9],grad.label_img,'haar')
p = rbepwt.Picture()
p.load_array(fdi)
p.show()
