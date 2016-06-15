import rbepwt

for k in range(45,55,1):
    i = rbepwt.Image()
    i.load('pickled/gradient')
    i.rbepwt.threshold_coefs(k)
    i.decode_rbepwt()
    psnr = i.psnr()
    i.save_decoded('bugtest/%s.png' % (str(k)+'--'+str(psnr)))
    print(k,psnr)
