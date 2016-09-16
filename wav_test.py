import rbepwt
import pywt
import timeit

outdir='wavelet_test/'
logfile = outdir+'log'
levels = 16
coefs = 512

wavelets = pywt.wavelist()
cc = rbepwt.Image()
cc.read('img/cameraman256.png')
#cc.read('img/gradient64.jpg')
cc.segment(scale=200,sigma=2,min_size=10)
for wav in wavelets:
    print("Encoding image with wavelet %s" % wav)
    start_time = timeit.default_timer()
    cc.encode_rbepwt(levels,wav,'easypath',euclidean_distance=True)
    time = timeit.default_timer() - start_time
    cc.save_pickle(outdir+'cameraman-encoded-'+wav)
    cc.rbepwt.threshold_coefs(coefs)
    cc.decode_rbepwt()
    cc.save_pickle(outdir+'cameraman-'+wav+str(coefs))
    cc.save_decoded(outdir+'cameraman-'+wav+str(coefs)+'image.png',title=None)
    psnr = cc.psnr()
    ssim = cc.ssim()
    vsi = cc.vsi()
    haarpsi = cc.haarpsi()
    logline = ",".join((wav,str(time),str(psnr),str(ssim),str(vsi),str(haarpsi)))
    #logline = wav+str(time)
    print(logline)
    outfile = open(logfile,'a')
    outfile.write(logline+'\n')
    outfile.close()
#outfile.close()

    
