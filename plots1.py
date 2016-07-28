import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

#load the table computed in table-characteristics notebook
table_file = open('decoded_pickles/table','rb')
table = pickle.load(table_file)
table_file.close

thresholds = [4096,2048,1024,512]

def error_plots():
    vecs = {}
    for index in ['psnr','ssim','vsi','haarpsi']:
        fig = plt.figure()
        #plt.style.use('classic')
        #
        plt.style.use('bmh')
        #plt.style.use('ggplot')
        #plt.style.use('seaborn-paper')
        plt.title(index)
        for imgname in ['cameraman','house','peppers']:
            for enc in ['easypath','gradpath','epwt-easypath','tensor']:
                vec = np.array(table[(table['image'] == imgname) & (table['encoding'] == enc)].sort('coefficients')[index])
                #print(imgname,enc,vec)
                if enc == 'easypath':
                    lsty = 'b'
                elif enc == 'gradpath':
                    lsty = 'r'
                elif enc == 'epwt-easypath':
                    lsty = 'g'
                elif enc == 'tensor':
                    lsty = 'k'
                if imgname == 'cameraman':
                    lsty += '-' 
                elif imgname == 'house':
                    lsty += '-.'
                elif imgname == 'peppers':
                    lsty += '--'
                #splinerange = np.arange(thresholds[-1],thresholds[0],(thresholds[0] - thresholds[-1])/10000)
                #spline = scipy.interpolate.spline(thresholds[::-1],vec,splinerange,order=2)
                #plt.plot(splinerange,spline,lsty,label=imgname+'-'+enc)
                plt.plot(thresholds[::-1],vec,lsty,label=imgname+'-'+enc)
        plt.legend(loc=4)
        #plt.yscale('log')
        plt.show()


if __name__ == '__main__':
    error_plots()
