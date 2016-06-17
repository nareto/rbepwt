import rbepwt
wav = 'bior4.4'
levels = 8
i = rbepwt.Image()
#i.read('img/cameraman.png')
i.read('img/gradient64.jpg')
#i.read('img/sampleimg4x4.png')
#i.segment(scale=2,sigma=0,min_size=1)
pickled_string='gradient64-%dlevels'%levels
i.segment(scale=200,sigma=0.8,min_size=10)
i.encode_rbepwt(levels,wav)
i.save('pickled/'+pickled_string)
for k in range(1,1000,10):
    i = rbepwt.Image()
    i.load('pickled/'+pickled_string)
    i.rbepwt.threshold_coefs(k)
    i.decode_rbepwt()
    psnr = i.psnr()
    i.save_decoded('bugtest/%s.png' % (str(k)+'--'+str(psnr)))
    print(k,psnr)
