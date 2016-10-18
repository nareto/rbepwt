import rbepwt
import pandas as pd
import matplotlib.pyplot as plt

epsilon = 2e-6
logfile='wavelet_test/log'
#logfile='wavelet_test-8levels/log'
#df = pd.read_csv(logfile,index_col=0,names=['wavelet','time','haarpsi'])
df = pd.read_csv(logfile,index_col=0,names=['wavelet','time','psnr','ssim','vsi','haarpsi'])
#series = df['haarpsi']
series = df.sort('haarpsi',ascending=False)['haarpsi']
#series = df.sort('haarpsi',ascending=False)[['haarpsi','ssim','vsi']]
series.plot(kind='bar')
minv = series.min()
maxv = series.max()
#minv = series['haarpsi'].min()
#maxv = series['haarpsi'].max()
plt.ylim(minv - epsilon, maxv + epsilon)
plt.yscale('log')
plt.show()
