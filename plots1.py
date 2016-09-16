import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import pickle
import rbepwt
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
### for Palatino and other serif fonts use:
##rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

thresholds = [4096,2048,1024,512]
imgpath = 'img/'
savedir = 'decoded_pickles-euclidean/'
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
    imgnames = ['cameraman256','peppers256','house256']
    #imgnames = ['peppers256','house256']
    #imgname = 'peppers256'
    #imgname = 'house256'
    #thresholds = [512,1024,2048,4096]
    thresholds = [512]

    for imgname in imgnames:
        #firstsave = True #change to False to save original image and segmentation
        firstsave = False
        for ncoefs in thresholds:
            #for enc in ['easypath','gradpath','epwt-easypath','tensor']:
            #for enc in ['easypath','gradpath']:
            for enc in ['easypath']:
                img = rbepwt.Image()
                if enc == 'tensor':
                    levs = '4'
                    loadstr = savedir+imgname+'-'+enc+'-bior4.4'+'-'+levs+'levels--'+str(ncoefs)
                else:
                    levs = '16'
                    loadstr = savedir+imgname+'-'+enc+'-bior4.4'+'-'+levs+'levels-euclidean--'+str(ncoefs)
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
        #for imgname in ['cameraman','house','peppers']:
        #for imgname in ['peppers']:
        #for imgname in ['cameraman']:
        for imgname in ['house']:
            for enc in ['easypath','gradpath','epwt-easypath','tensor']:
                vec = np.array(table[(table['image'] == imgname) & (table['encoding'] == enc)].sort('coefficients')[index])
                #print(imgname,enc,vec)
                mrkr = None
                if enc == 'easypath':
                    #lsty = '-'
                    lsty = 'None'
                    mrkr = '_'
                elif enc == 'gradpath':
                    #lsty = '-.'
                    lsty = 'None'
                    mrkr = '|'
                elif enc == 'epwt-easypath':
                    #lsty = '--'
                    lsty = 'None'
                    mrkr = 'x'
                elif enc == 'tensor':
                    #lsty = ':'
                    lsty = 'None'
                    mrkr = '.'
                if imgname == 'cameraman':
                    lco = 'b'
                elif imgname == 'house':
                    lco = 'r'
                elif imgname == 'peppers':
                    lco = 'g'
                #lab = imgname+'-'+enc
                lab = enc
                if enc == 'epwt-easypath':
                    lab = 'epwt'
                if mrkr is not None:
                    ax.plot(thresholds[::-1],vec,color=lco,linestyle=lsty,marker=mrkr,markersize=12.0,markeredgewidth=2,alpha=0.8)
                else:
                    ax.plot(thresholds[::-1],vec,color=lco,linestyle=lsty)                    
                ax.set_xticks(thresholds)
                ax.set_xlim(420,5000)
                #labels.append(lab)
                #if enc == 'easypath' and index =='psnr':
                #if imgname == 'cameraman' and index == 'psnr':
                if imgname == 'house' and index == 'psnr':
                    #handles.append(mlines.Line2D([], [], color='k', linestyle=lsty,#marker=lsty[1:],
                    #                             markersize=15, label=lab))
                    #handles += ax.plot([], [], color='k',linestyle=lsty, label=enc)
                    #handles.append(mlines.Line2D([], [], color='k',linestyle=lsty, label=imgname))
                    handles.append(mlines.Line2D([], [], color='k',linestyle=lsty,label=lab,marker=mrkr))
    #plt.figlegend(tuple(lines),tuple(labels),loc='lower left',ncol=4,labelspacing=0.2)
    #plt.figlegend((lines[0],lines[1]),(labels[0],labels[1]),loc='lower left',ncol=4,labelspacing=0.2)
    #ax5.legend(lines[0],labels[0],loc='lower left')#,ncol=4,labelspacing=0.2)
    #co_easypath = mpatches.Patch(color='blue', label='Easypath')
    #co_gradpath = mpatches.Patch(color='red', label='Gradpath')
    #co_epwt = mpatches.Patch(color='green', label='EPWT')
    #co_tensor = mpatches.Patch(color='k', label='Tensor')
    #handles = [co_easypath,co_gradpath,co_epwt,co_tensor] + handles
    co_cam = mpatches.Patch(color='blue', label='Cameraman')
    co_house = mpatches.Patch(color='red', label='House')
    co_pep = mpatches.Patch(color='green', label='Peppers')
    #handles = [co_cam,co_pep,co_house] + handles
    #handles = [co_cam] + handles
    #handles = [co_pep] + handles
    handles = [co_house] + handles
    #ax5.axis('off')
    #ax6.axis('off')
    #ax5.legend(lines[0],labels[0],loc='center')#,ncol=4,labelspacing=0.2)
    #ax5.legend(handles=lines,labels=labels,loc='center',ncol=1,labelspacing=0.2)
    #ax5.legend(handles=handles,loc='center',ncol=1,labelspacing=0.2)
    plt.legend(handles=handles,loc='lower right',ncol=1,labelspacing=0.2)
    #ax.yscale('log')
    plt.tight_layout()
    #fig.savefig(export_dir+'errors.png')
    #fig.show()
    plt.show()

def error_plots2(table):
    vecs = {}
    msize=12
    fsize=12
    for imgname in ['cameraman','house','peppers']:
    #for imgname in ['cameraman']:
        #fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(18,9))
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,6))
        #plt.style.use('bmh')
        plt.title(imgname)
        #plt.rcParams['figure.figsize'] = (24, 12)
        axes = [ax1,ax2]
        handles = []
        labels = []
        for i,index in enumerate(['psnr','haarpsi']):
            #plt.style.use('classic')
            ax = axes[i]
            ax.tick_params(axis='both', labelsize=fsize)
            ax.grid(True)
            ax.set_title(index,fontdict={'fontsize': fsize})
            for enc in ['easypath','gradpath','epwt-easypath','tensor']:
                vec = np.array(table[(table['image'] == imgname) & (table['encoding'] == enc)].sort('coefficients')[index])
                #print(imgname,enc,vec)
                mrkr = None
                if enc == 'easypath':
                    #lsty = '-'
                    lsty = 'None'
                    mrkr = '_'
                    lco = 'g'
                elif enc == 'gradpath':
                    #lsty = '-.'
                    lsty = 'None'
                    mrkr = '|'
                    lco = 'r'
                elif enc == 'epwt-easypath':
                    #lsty = '--'
                    lsty = 'None'
                    mrkr = 'x'
                    lco = 'b'
                elif enc == 'tensor':
                    #lsty = ':'
                    lsty = 'None'
                    lco = 'k'
                    mrkr = '.'
                lab = enc
                if enc == 'epwt-easypath':
                    lab = 'epwt'
                if mrkr is not None:
                    ax.plot(thresholds[::-1],vec,color=lco,linestyle=lsty,marker=mrkr,markersize=msize,markeredgewidth=2)
                else:
                    ax.plot(thresholds[::-1],vec,color=lco,linestyle=lsty)                    
                ax.set_xticks(thresholds)
                ax.set_xlim(420,5000)
                #labels.append(lab)
                #if enc == 'easypath' and index =='psnr':
                #if imgname == 'cameraman' and index == 'psnr':
                if index == 'haarpsi':
                    handles.append(mlines.Line2D([], [], color=lco,linestyle=lsty,label=lab,marker=mrkr))
        leg = plt.legend(handles=handles,loc='lower right',ncol=1,labelspacing=0.2,fontsize='xx-large',markerscale=5,numpoints=1)
        #for l in leg.get_lines():
        #    l.set_linewidth(20)
        #ax.yscale('log')
        plt.tight_layout()
        #fig.savefig(export_dir+'errors.png')
        fig.savefig(export_dir+'errors-'+imgname+'.png')
        #fig.show()
        #plt.show()


if __name__ == '__main__':
    table = load_table('decoded_pickles-euclidean/table')
    error_plots2(table)
    #decoded_plots(table,True)
