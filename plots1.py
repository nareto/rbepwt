import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
    #fig, ((ax1, ax2), (ax3, ax4), (ax5,ax6)) = plt.subplots(3, 2)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    plt.style.use('bmh')
    plt.rcParams['figure.figsize'] = (12, 12)
    axes = [ax1,ax2,ax3,ax4]
    handles = []
    labels = []
    for i,index in enumerate(['psnr','ssim','vsi','haarpsi']):
        #fig = plt.figure()
        #plt.style.use('classic')
        ax = axes[i]
        #ax.style.use('ggplot')
        #ax.style.use('seaborn-paper')
        ax.set_title(index)
        for imgname in ['cameraman','house','peppers']:
            for enc in ['easypath','gradpath','epwt-easypath','tensor']:
                vec = np.array(table[(table['image'] == imgname) & (table['encoding'] == enc)].sort('coefficients')[index])
                #print(imgname,enc,vec)
                if enc == 'easypath':
                    lco = 'b'
                elif enc == 'gradpath':
                    lco = 'r'
                elif enc == 'epwt-easypath':
                    lco = 'g'
                elif enc == 'tensor':
                    lco = 'k'
                if imgname == 'cameraman':
                    lsty = '-' 
                elif imgname == 'house':
                    lsty = '-.'
                elif imgname == 'peppers':
                    lsty = '--'
                lab = imgname+'-'+enc
                ax.plot(thresholds[::-1],vec,color=lco,linestyle=lsty)                    
                ax.set_xticks(thresholds)
                #labels.append(lab)
                if enc == 'easypath' and index =='psnr':
                    #handles.append(mlines.Line2D([], [], color='k', linestyle=lsty,#marker=lsty[1:],
                    #                             markersize=15, label=lab))
                    #handles += ax.plot([], [], color='k',linestyle=lsty, label=enc)
                    handles.append(mlines.Line2D([], [], color='k',linestyle=lsty, label=imgname))
    #plt.figlegend(tuple(lines),tuple(labels),loc='lower left',ncol=4,labelspacing=0.2)
    #plt.figlegend((lines[0],lines[1]),(labels[0],labels[1]),loc='lower left',ncol=4,labelspacing=0.2)
    #ax5.legend(lines[0],labels[0],loc='lower left')#,ncol=4,labelspacing=0.2)
    co_easypath = mpatches.Patch(color='blue', label='Easypath')
    co_gradpath = mpatches.Patch(color='red', label='Gradpath')
    co_epwt = mpatches.Patch(color='green', label='EPWT')
    co_tensor = mpatches.Patch(color='k', label='Tensor')
    handles = [co_easypath,co_gradpath,co_epwt,co_tensor] + handles
    #ax5.axis('off')
    #ax6.axis('off')
    #ax5.legend(lines[0],labels[0],loc='center')#,ncol=4,labelspacing=0.2)
    #ax5.legend(handles=lines,labels=labels,loc='center',ncol=1,labelspacing=0.2)
    #ax5.legend(handles=handles,loc='center',ncol=1,labelspacing=0.2)
    plt.legend(handles=handles,loc='lower right',ncol=1,labelspacing=0.2)
    #ax.yscale('log')
    plt.tight_layout()
    fig.savefig(export_dir+'errors.png')
    #fig.show()
    #plt.show()


if __name__ == '__main__':
    table = load_table('decoded_pickles-euclidean/table')
    error_plots(table)
    #decoded_plots(table,True)
