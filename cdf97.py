import rbepwt
import numpy as np
import pywt

wav = rbepwt.cdf97
#wav='sym2'
mode = 'periodization'
#mode = 'symmetric'
z= np.arange(8)
a,d = pywt.dwt(z,wav,mode)
print("approx: %s \n detail: %s \n" %(a,d))
print(pywt.idwt(a,d,wav,mode))
