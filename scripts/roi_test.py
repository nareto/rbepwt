import rbepwt



#i,j,I,J where i and j are coords for top left point and I and J for bottom right
rect = (40,90, 110,150)

def simple_roi():
    img1 = rbepwt.Image()
    #img.load_pickle('../decoded_pickles-euclidean/cameraman256-easypath-bior4.4-16levels--512')
    #img.load_pickle('../decoded_pickles-euclidean/peppers256-easypath-bior4.4-16levels--512')
    #img.load_pickle('../pickled/cameraman256-easypath-bior4.4-16levels')
    #img1.load_pickle('../pickled/gradient64-easypath-bior4.4-12levels')
    img1.load_pickle('../pickled/cameraman256-easypath-haar-16levels')
    roi = rbepwt.Roi(img1)

    #regions = roi.find_intersecting_regions(img,rect)
    regions= set([1])
    roi.compute_roi_coeffs(regions)
    #img1.threshold_coefs(512)
    img1.decode_rbepwt()
    img1.show_decoded(title=None)
 
    #roi.compute_roi_coeffs(img2,regions)
    #img2.threshold_coefs(512)
    #img2.decode_rbepwt()
    #img2.show_decoded(title='RBEPWT at first level only')


def in_out_roi():
    img1 = rbepwt.Image()
    #img.load_pickle('../decoded_pickles-euclidean/cameraman256-easypath-bior4.4-16levels--512')
    #img.load_pickle('../decoded_pickles-euclidean/peppers256-easypath-bior4.4-16levels--512')
    #img.load_pickle('../pickled/cameraman256-easypath-bior4.4-16levels')
    #img1.load_pickle('../pickled/gradient64-easypath-bior4.4-12levels')
    img1.load_pickle('../pickled/cameraman256-easypath-haar-16levels')
    roi = rbepwt.Roi(img1)

    #regions = roi.find_intersecting_regions(img,rect)
    regions= set([1,7])
    roi.compute_dual_roi_coeffs(regions,0.5,0.01)
    #img1.threshold_coefs(512)
    img1.decode_rbepwt()
    img1.show_decoded(title=None)
    return(img1)

if __name__ == '__main__':
    out = in_out_roi()
    

