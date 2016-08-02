import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import rbepwt

thresholds = [4096,2048,1024,512]
imgpath = 'img/'
savedir = 'decoded_pickles/'
export_dir = '/Users/renato/ownCloud/phd/talks-papers/rbepwt-canazeiproceedings/img/'

def load_table(pickle_path):
    #load the table computed in table-characteristics notebook
    table_file = open(pickle_path,'rb')
    table = pickle.load(table_file)
    table_file.close
    return(table)


def recompute_table():
    table = pd.DataFrame(columns=['image','encoding','wavelet','levels','coefficients','psnr','ssim','vsi','haarpsi'])
    img_names = ['cameraman256','house256','peppers256']
    #img_names = ['peppers256']
    encodings = ['easypath','gradpath','epwt-easypath','tensor']
    #encodings = ['gradpath','tensor']
    for thresh in thresholds:
        for imgname in img_names:
            for enc in encodings:
                print("working on %s with encoding %s and threshold %d" % (imgname,enc,thresh))
                img = rbepwt.Image()
                if enc == 'tensor':
                    levs = '4'
                    #levs = '8'
                else:
                    levs = '16'
                loadstr = savedir+imgname+'-'+enc+'-bior4.4'+'-'+levs+'levels--'+str(thresh)
                #loadstr = savedir+imgname+'-'+enc+'-haar'+'-'+levs+'levels--'+str(thresh)
                #print(loadstr)
                img.load_pickle(loadstr)
                psnr = img.psnr()
                ssim = img.ssim()
                vsi = img.vsi()
                haarpsi = img.haarpsi()
                table.loc[len(table)] = [imgname.rstrip('256'),enc,'bior4.4',int(levs),thresh,psnr,ssim,vsi,haarpsi]
            #table.loc[len(table)] = [imgname.rstrip('256'),enc,'haar',int(levs),thresh,psnr,ssim,vsi,haarpsi]    


def decoded_plots(table,save=False):
    imgname = 'cameraman256'
    #imgname = 'peppers256'
    #imgname = 'house256'
    ncoefs = 1024
    firstsave = False
    for enc in ['easypath','gradpath','epwt-easypath','tensor']:
        img = rbepwt.Image()
        if enc == 'tensor':
            levs = '4'
        else:
            levs = '16'
        loadstr = savedir+imgname+'-'+enc+'-bior4.4'+'-'+levs+'levels--'+str(ncoefs)
        img.load_pickle(loadstr)
        if save:
            fname = "-".join((imgname,enc,'bior4.4',str(ncoefs),levs))+'.png'
            img.save_decoded(export_dir+fname,title=None)
            if not firstsave:
                firstsave = True
                orig_fname = imgname+'.png'
                img.save(export_dir+orig_fname)
                seg_fname="-".join((imgname,"segmentation"))+'.png'
                img.save_segmentation(title=None,filepath=export_dir+seg_fname)
        else:
            img.show_decoded(title='')


def error_plots(table):
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
    table = load_table('decoded_pickles/table')
    #error_plots(table)
    decoded_plots(table,True)
