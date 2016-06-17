    import numpy as np
    import pywt
    
    factor = 1
    cdf97_an_lo = factor*np.array([0.026748757411, -0.016864118443, -0.078223266529, 0.266864118443,\
                            0.602949018236,	0.266864118443,	-0.078223266529,-0.016864118443, \
                            0.026748757411])
    cdf97_an_hi = factor*np.array([0, 0.091271763114, -0.057543526229,-0.591271763114,1.11508705,\
                            -0.591271763114,-0.057543526229,0.091271763114,0 ])
    cdf97_syn_lo = factor*np.array([0,-0.091271763114,-0.057543526229,0.591271763114,1.11508705,\
                             0.591271763114	,-0.057543526229,-0.091271763114,0])
    cdf97_syn_hi = factor*np.array([0.026748757411,0.016864118443,-0.078223266529,-0.266864118443,\
                             0.602949018236,-0.266864118443,-0.078223266529,0.016864118443,\
                             0.026748757411])
    cdf97 = pywt.Wavelet('cdf97', [cdf97_an_lo,cdf97_an_hi,cdf97_syn_lo,cdf97_syn_hi])
    wav = cdf97
    mode = 'periodization'
    sig= np.arange(8)
    a,d = pywt.dwt(sig,wav,mode)
    print("approx: %s \n detail: %s \n" %(a,d))
    rec_sig = pywt.idwt(a,d,wav,mode)
    print(rec_sig)
    
print(cdf97.get_filters_coeffs()[0])
