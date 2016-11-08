import rbepwt
import roi


#i,j,I,J where i and j are coords for top left point and I and J for bottom right
rect = (40,90, 110,150)

img = rbepwt.Image()
#img.load_pickle('../decoded_pickles-euclidean/cameraman256-easypath-bior4.4-16levels--512')
#img.load_pickle('../decoded_pickles-euclidean/peppers256-easypath-bior4.4-16levels--512')
#img.load_pickle('../pickled/cameraman256-easypath-bior4.4-16levels')
img.load_pickle('../pickled/cameraman256-easypath-haar-16levels')

#regions = roi.find_intersecting_regions(img,rect)
regions= set([7])
roi.compute_roi_coeffs(img,regions)
#img.threshold_coefs(512)
img.decode_rbepwt()
img.show_decoded()

