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
        self.roi_border = []
        self.press = False
        self.interior_point = None
        self.draw_points = draw_points
        self.border_is_closed = False
        if draw_points:
            self.roi_border_plt, = plt.plot([],[],'sr')
        else:
            self.roi_border_plt = [plt.plot([],[],'-r')[0]]

    def draw_roi(self,points = False):
        if points:
            xdata = [x[0] for x in self.roi_border]
            ydata = [x[1] for x in self.roi_border]
            self.roi_border_plt.set_xdata(xdata)
            self.roi_border_plt.set_ydata(ydata)
        else:
            prev_point = self.roi_border[0]
            for idx,point in enumerate(self.roi_border[1:]):
                cur_point = point
                self.roi_border_plt[idx].set_xdata([prev_point[0],cur_point[0]])
                self.roi_border_plt[idx].set_ydata([prev_point[1],cur_point[1]])
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
        print('Draw a closed curve in the image')

        
    def disconnect(self):
        'disconnect all the stored connection ids'
        self.fig.canvas.mpl_disconnect(self.cidpress)
        self.fig.canvas.mpl_disconnect(self.cidrelease)
        self.fig.canvas.mpl_disconnect(self.cidmotion)


    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.axes: return
        self.press = True
        point = np.array((int(event.xdata),int(event.ydata)))
        self.roi_border.append(point)

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if not self.press: return
        if event.inaxes != self.axes: return
        #print((int(event.xdata),int(event.ydata)))
        point = np.array((int(event.xdata),int(event.ydata)))
        if np.array_equal(point,self.roi_border[-1]): return
        if self.press:
            target_point = point
            cur_point = self.roi_border[-1]
            while not np.array_equal(target_point,cur_point):
                neigh = rbepwt.neighborhood(cur_point,1,mode='square',hole=True)
                min_dist = np.linalg.norm(target_point-cur_point)
                for p in neigh[1:]:
                    p = np.array(p)
                    d = np.linalg.norm(target_point - p)
                    if d < min_dist:
                        min_dist = d
                        cur_point = p
                if is_array_in_list(cur_point,self.roi_border) and len(self.roi_border) > 3:
                    self.border_is_closed = True
                    self.draw_roi()
                    self.__second_routine__()
                    return
                self.roi_border.append(cur_point)
                if not self.draw_points:
                    self.roi_border_plt.append(plt.plot([],[],'-r')[0])
            self.draw_roi()


    def on_release(self,event):
        self.press = False
        self.__second_routine__()

    def __second_routine__(self):
        self.disconnect()
        if not self.border_is_closed: return
        print('Click on a point in the interior of the curve')
        self.choose_point()
        
    def __third_routine__(self):
        print('Close the window')
    
    def draw(self):
        """Opens a window where the user should draw (click+drag) a closed curve and returns the set of points in the interior"""
        plt.imshow(self.array,cmap=plt.cm.gray)
        plt.xlim((0,self.array.shape[1]))
        plt.ylim((self.array.shape[0],0))
        plt.show()

    def choose_point(self):
        def cid_choose_point(event):
            self.interior_point = (int(event.xdata),int(event.ydata))
            self.fig.canvas.mpl_disconnect(self.cidpress)
            self.__find_region__()
            self.__third_routine__()
        self.cidpress = self.fig.canvas.mpl_connect(
            'button_press_event', cid_choose_point)

    def draw_border(self):
        mat = np.zeros_like(self.array)
        for p in self.roi_border:
            mat[p[1],p[0]] = 1
        plt.imshow(mat,cmap=plt.cm.gray)
        plt.show()
        
    def __find_region__(self):
        border = set()
        for arr in self.roi_border:
            border.add(tuple(arr))
        #ret = np.zeros_like(self.array)
        next_points = set()
        next_points.add(self.interior_point)
        region_points = set()
        region_points.add(self.interior_point)
        found_all = False
        m,n = self.array.shape
        while next_points:
            cur_point = next_points.pop()
            neigh = rbepwt.neighborhood(cur_point,1,mode='cross',hole=True)
            for p in neigh:
                if p[0] >= 0 and p[0] < n and p[1] >=0 and p[1] < m and p not in border and p not in region_points:
                    region_points.add(p)
                    if p not in next_points:
                        next_points.add(p)
        self.region_points = region_points
        return(region_points)

        
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

    region_points = list(drawable.region_points)
    values = []
    regions = {}
    for p in region_points:
        point = (p[1],p[0])
        value = img[point]
        lab = img.segmentation.label_img[point]
        if lab in regions.keys():
            regions[lab].add_point(point,value)
        else:
            regions[lab] = rbepwt.Region([point],[value])

    region_collection = rbepwt.RegionCollection(*list(regions.values()))
    #print(region.base_points)
    #region.show(px_value=True)
    return(region_collection)
    
if __name__ == '__main__':
    #out1,out2 = in_out_roi(1,0,False)
    #out1,out2 = in_out_roi(1,0,False)
    #out1,out2 = in_out_roi(0.1,0.001)
    #simple_roi()
    region_collection = select_roi()
    

