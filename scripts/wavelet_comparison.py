#    Copyright 2017 Renato Budinich
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

#Here we test the performance of different wavelets for our method.

import rbepwt
import pandas as pd
import matplotlib.pyplot as plt
import timeit

def encode_and_write_log():
    outdir='../wavelet_test/'
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

def draw_plot():
    epsilon = 2e-6
    logfile='../wavelet_test/log'
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


if __name__ == '__main__':
    draw_plot()
