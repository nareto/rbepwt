import rbepwt



#i,j,I,J where i and j are coords for top left point and I and J for bottom right
rect = (40,90, 110,150)

def simple_roi():
    img1 = rbepwt.Image()
    #img.load_pickle('../decoded_pickles-euclidean/cameraman256-easypath-bior4.4-16levels--512')
    #img.load_pickle('../decoded_pickles-euclidean/peppers256-easypath-bior4.4-16levels--512')
    #img.load_pickle('../pickled/cameraman256-easypath-bior4.4-16levels')
    #img1.load_pickle('../pickled/gradient64-easypath-bior4.4-12levels')
    #img1.load_pickle('../pickled/cameraman256-easypath-haar-16levels')
    picklepath = '../pickled/house256-easypath-haar-16levels'
    #picklepath = '../pickled/sampleimg-easypath-haar-4levels'
    img1.load_pickle(picklepath)
    roi = rbepwt.Roi(img1)

    #regions = roi.find_intersecting_regions(img,rect)
    #regions= set([1,2,3,5])#cameraman?
    #regions = set([18,15,7,19])
    regions = set([7])
    #regions = set([4])
    roi.compute_roi_coeffs(regions)
    #img1.threshold_coefs(512)
    img1.decode_rbepwt()
    img1.show_decoded(title=None)
    #print(img1.decoded_img)
    #roi.compute_roi_coeffs(img2,regions)
    #img2.threshold_coefs(512)
    #img2.decode_rbepwt()
    #img2.show_decoded(title='RBEPWT at first level only')
    return(img1)

def in_out_roi(percin,percout,second_image=True):
    #picklepath = '../pickled/cameraman256-easypath-haar-16levels'
    picklepath = '../pickled/house256-easypath-haar-16levels'
    #img.load_pickle('../decoded_pickles-euclidean/cameraman256-easypath-bior4.4-16levels--512')
    #img.load_pickle('../decoded_pickles-euclidean/peppers256-easypath-bior4.4-16levels--512')
    #img.load_pickle('../pickled/cameraman256-easypath-bior4.4-16levels')
    #img1.load_pickle('../pickled/gradient64-easypath-haar-12levels')
    img1 = rbepwt.Image()
    img2 = rbepwt.Image()
    img1.load_pickle(picklepath)
    img2.load_pickle(picklepath)
    roi = rbepwt.Roi(img1)

    #regions = roi.find_intersecting_regions(img,rect)
    #regions= set([1,2,3,5])
    regions= set([18,15,7,19])
    #img1.mask_region(regions)
    #regions= set([1,3])
    nin,nout = roi.compute_dual_roi_coeffs(regions,percin,percout)
    #img1.threshold_coefs(512)
    img1.decode_rbepwt()
    totcoefs = img1.nonzero_coefs()
    img1.show_decoded(title='tot coefs = %4d, in = %3f (%d), out = %3f (%d)' % (totcoefs,percin,nin,percout,nout) )

    if second_image:
        img2.threshold_coefs(totcoefs)
        img2.decode_rbepwt()
        img2.show_decoded(title='tot coefs = %4d' % img2.nonzero_coefs())
    return(img1,img2)

if __name__ == '__main__':
    #out1,out2 = in_out_roi(1,0,False)
    #out1,out2 = in_out_roi(1,0,False)
    #out1,out2 = in_out_roi(0.1,0.001)
    out = simple_roi()
    

