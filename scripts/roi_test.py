import ipdb
import rbepwt


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



def simple_roi():
    picklepath = '../pickled/cameraman256-easypath-haar-16levels'
    img = rbepwt.Image()
    img.load_pickle(picklepath)
    drawable = rbepwt.DrawRoi(img)
    region_collection = drawable.main()
    region_collection.encode_rbepwt(4,'bior4.4')
    #nzcoefs = len(region_collection.points)
    nzcoefs = region_collection.nonzero_coefs()
    nthresh = 0.1*nzcoefs
    region_collection.threshold_coefs(nthresh)
    print('nonzero coefs = %d, after thresholding = %d in theory, and %d in practice' % (nzcoefs,nthresh,region_collection.nonzero_coefs()))
    
    region_collection.decode_rbepwt()
    region_collection.show()
    region_collection.decoded_region_collection.show()
    return(region_collection)
    
if __name__ == '__main__':
    #out1,out2 = in_out_roi(1,0,False)
    #out1,out2 = in_out_roi(1,0,False)
    #out1,out2 = in_out_roi(0.1,0.001)
    #simple_roi()
    region_collection = simple_roi()
    

