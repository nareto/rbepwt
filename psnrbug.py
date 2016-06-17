import rbepwt

for k in range(1,500,1):
    i = rbepwt.Image()
    i.load('pickled/gradient64-8levels')
    i.rbepwt.threshold_coefs(k)
    i.decode_rbepwt()
    psnr = i.psnr()
    i.save_decoded('bugtest/%s.png' % (str(k)+'--'+str(psnr)))
    print(k,psnr)
