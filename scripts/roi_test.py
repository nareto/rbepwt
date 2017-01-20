import ipdb
import rbepwt


def is_array_in_list(array,l):
    for x in l:
        if np.array_equal(x,array):
            return(True)
    return(False)

class DrawRoi:
    
    def __init__(self,array,draw_points=False):
        self.array = array
        self.fig = plt.figure()
        self.axes = self.fig.gca()
        self.axes.invert_yaxis()
        self.roi = []
        self.press = False
        self.draw_points = draw_points
        if draw_points:
            self.roi_plt, = plt.plot([],[],'sr')
        else:
            self.roi_plt = [plt.plot([],[],'-r')[0]]

    def draw_roi(self,points = False):
        if points:
            xdata = [x[0] for x in self.roi]
            ydata = [x[1] for x in self.roi]
            self.roi_plt.set_xdata(xdata)
            self.roi_plt.set_ydata(ydata)
        else:
            prev_point = self.roi[0]
            for idx,point in enumerate(self.roi[1:]):
                cur_point = point
                self.roi_plt[idx].set_xdata([prev_point[0],cur_point[0]])
                self.roi_plt[idx].set_ydata([prev_point[1],cur_point[1]])
                prev_point = cur_point
        plt.draw()
        
    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidmotion = self.fig.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cidrelease = self.fig.canvas.mpl_connect(
            'button_release_event', self.on_release)

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.fig.canvas.mpl_disconnect(self.cidpress)
        self.fig.canvas.mpl_disconnect(self.cidrelease)
        self.fig.canvas.mpl_disconnect(self.cidmotion)
        #for idx,arr in enumerate(self.roi):
        #    print('%4d  %s' % (idx,tuple(arr)))

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.axes: return
        self.press = True
        print(event)
        point = np.array((int(event.xdata),int(event.ydata)))
        self.roi.append(point)

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if not self.press: return
        if event.inaxes != self.axes: return
        #print((int(event.xdata),int(event.ydata)))
        point = np.array((int(event.xdata),int(event.ydata)))
        if np.array_equal(point,self.roi[-1]): return
        if self.press:
            target_point = point
            cur_point = self.roi[-1]
            while not np.array_equal(target_point,cur_point):
                neigh = rbepwt.neighborhood(cur_point,1,mode='square',hole=True)
                min_dist = np.linalg.norm(target_point-cur_point)
                for p in neigh[1:]:
                    p = np.array(p)
                    d = np.linalg.norm(target_point - p)
                    if d < min_dist:
                        min_dist = d
                        cur_point = p
                if is_array_in_list(cur_point,self.roi) and len(self.roi) > 3:
                    self.disconnect()
                    self.draw_roi()
                    return
                self.roi.append(cur_point)
                if not self.draw_points:
                    self.roi_plt.append(plt.plot([],[],'-r')[0])
            self.draw_roi()


    def on_release(self,event):
        self.press = False
        self.disconnect()
        
    def draw(self):
        """Opens a window where the user should draw (click+drag) a closed curve and returns the set of points in the interior"""
        plt.imshow(self.array,cmap=plt.cm.gray)
        plt.xlim((0,self.array.shape[1]))
        plt.ylim((self.array.shape[0],0))
        plt.show()

    def find_region(self):
        border = set()
        for arr in self.roi:
            border.add(tuple(arr))
        ret = np.ones_like(self.array)
        cur_point = (0,0)
        found = False
        while not found:
            neigh = rbepwt.neighborhood(cur_point,1,mode='square',hole=True)
            for p in neigh:
                m,n = self.array.shape
                if p[0] >= 0 and p[0] < n and p[1] >=0 and p[1] < m and p not in border:
                    ret[p] = 0
            

        
def in_out_roi(percin,percout,second_image=True):
    #picklepath = '../pickled/cameraman256-easypath-haar-16levels'
    picklepath = '../pickled/house256-easypath-haar-16levels'
    #img.load_pickle('../decoded_pickles-euclidean/cameraman256-easypath-bior4.4-16levels--512')
    #img.load_pickle('../decoded_pickles-euclidean/peppers256-easypath-bior4.4-16levels--512')
    #img.load_pickle('../pickled/cameraman256-easypath-bior4.4-16levels')
    #img1.load_pickle('../pickled/gradient64-easypath-haar-12levels')
    img1 = rbepwt.Image()
    img2 = rbepwt.Image()
    img1.load_pickle(picklepath)
    img2.load_pickle(picklepath)
    roi = rbepwt.Roi(img1)

    #regions = roi.find_intersecting_regions(img,rect)
    #regions= set([1,2,3,5])
    regions= set([18,15,7,19])
    #img1.mask_region(regions)
    #regions= set([1,3])
    nin,nout = roi.compute_dual_roi_coeffs(regions,percin,percout)
    #img1.threshold_coefs(512)
    img1.decode_rbepwt()
    totcoefs = img1.nonzero_coefs()
    img1.show_decoded(title='tot coefs = %4d, in = %3f (%d), out = %3f (%d)' % (totcoefs,percin,nin,percout,nout) )

    if second_image:
        img2.threshold_coefs(totcoefs)
        img2.decode_rbepwt()
        img2.show_decoded(title='tot coefs = %4d' % img2.nonzero_coefs())
    return(img1,img2)

def draggable_rectangles():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects = ax.bar(range(10), 20*np.random.rand(10))
    drs = []
    for rect in rects:
        dr = DraggableRectangle(rect)
        dr.connect()
        drs.append(dr)

    plt.show()
    #picklepath = '../pickled/cameraman256-easypath-haar-16levels'
    #img = rbepwt.Image()
    #img.load_pickle(picklepath)
    #
    #z = img.img
    #fig = plt.figure()
    #ax = fig.gca()
    ##ax = fig.add_subplot(111)
    ##ax.plot(np.random.rand(10))
    #ax.imshow(z)
    #def onclick(event):
    #    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #          (event.button, event.x, event.y, event.xdata, event.ydata))
    #
    #cid = fig.canvas.mpl_connect('button_press_event', onclick)
    #plt.show()

def select_roi():
    picklepath = '../pickled/cameraman256-easypath-haar-16levels'
    img = rbepwt.Image()
    img.load_pickle(picklepath)
    drawable = DrawRoi(img.img)
    drawable.connect()
    drawable.draw()
    
if __name__ == '__main__':
    #out1,out2 = in_out_roi(1,0,False)
    #out1,out2 = in_out_roi(1,0,False)
    #out1,out2 = in_out_roi(0.1,0.001)
    #simple_roi()
    select_roi()
    

