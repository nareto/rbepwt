import rbepwt
import numpy as np
import ipdb

def random_walk_region(npoints):
    start_point = (0,0)
    random_walk = [start_point]
    while len(random_walk) < npoints:
        #cur_point = random_walk[-1]
        level = 1
        found = False
        while not found:
            cur_point = random_walk[np.random.randint(0,len(random_walk))]
            neighbors = rbepwt.neighborhood(cur_point,level,mode='cross',hole=True)
            #idx = np.random.randint(0,len(neighbors))
            np.random.shuffle(neighbors)
            for neigh in neighbors:
                if neigh in random_walk:
                    continue
                else:
                    random_walk.append(neigh)
                    found = True
                    break
            #level += 1
    return(random_walk)

def generate_easypath():
    random_walk = random_walk_region(40)
    #random_walk = ((0, 0), (0, 1), (0, -1), (1, 1), (1, 0), (1, -1), (1, 2), (2, 1), (2, 0), (-1, 0))
    region = rbepwt.Region(random_walk)
    #region.show(title='Original')
    region.pprint()

    easypath = region.easy_path(level=None)
    #easypath.show(True,title='EasyPath')
    #easypath.show(True)
    #easypath.show(True,px_value=0.5,path_color='green',border_thickness=0.02)
    easypath.show(True,path_color='black',border_thickness=0.02)
    easypath.pprint()

    #REDUCE:
    #newr = easypath.reduce_points()
    #easy_newr = newr.easy_path(level=None)
    #easy_newr.show(True,px_value=0.5,path_color='green',border_thickness=0.02)
    #easy_newr.show(True,path_color='green',border_thickness=0.02)

    #MAXDIST:
    md_easypath = region.easy_path(level=None,euclidean_distance=False)
    #easypath.show(True,title='EasyPath')
    #easypath.show(True)
    #easypath.show(True,px_value=0.5,path_color='green',border_thickness=0.02)
    md_easypath.show(True,path_color='black',border_thickness=0.02)
    
def generate_gradpath():
    #random_walk = random_walk_region(40)
    random_walk =  ((-3, -1), (-3, 0), (-2, 0), (-2, 1), (-2, 2), (-2, 3), (-1, 3), (-1, 4), (-1, 5), (0, 5), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0), (0, -1), (0, -2), (0, -3), (-1, -3), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (1, 2), (1, 3), (1, 4), (2, 2), (2, 1), (2, 0), (2, -1), (1, -1), (1, 0), (1, 1), (3, 1), (-2, -1), (-2, -2), (-3, 2), (-3, 3), (-2, 5))


    values1 = []
    values2 = []
    region = rbepwt.Region(random_walk)
    region.pprint()
    #topleft,bottomright = region.top_left,region.bottom_right
    ioffset,joffset = region.top_left
    shifted_random_walk = []
    for coord in random_walk:
        shifted_random_walk.append((coord[0] - ioffset,coord[1]-joffset))
    rows,cols = 1 + region.bottom_right[0]-region.top_left[0], 1 + region.bottom_right[1] - region.top_left[1]
    mat1 = np.zeros((rows,cols))
    mat2 = np.zeros((rows,cols))
    minj,mini,maxj,maxi = 0,0,cols-1,rows-1
    for coord,zero in np.ndenumerate(mat1):
        i,j = coord
        mat1[coord] = 200*(j/(maxj-minj) - minj/(maxj-minj))
        mat2[coord] = 200*(i/(maxi-mini) - mini/(maxi-mini))
    grad1 = np.gradient(mat1)
    grad2 = np.gradient(mat2)
    for coord in shifted_random_walk:
        values1.append(mat1[coord])
        values2.append(mat2[coord])
    r1 = rbepwt.Region(random_walk,values1)
    r1.compute_avg_gradient(grad1)
    r2 = rbepwt.Region(random_walk,values2)
    r2.compute_avg_gradient(grad2)

    g1 = r1.grad_path(level=None)
    g1.show(True,px_value=True,path_color='green')
    g1.pprint()

    g2 = r2.grad_path(level=None)
    g2.show(True,px_value=True,path_color='green')
    g2.pprint()

def generate_epwt():
    n = 4
    mat = 255*np.random.random((n,n))
    im = rbepwt.Image()
    im.read_array(mat)
    im.encode_epwt(1,'haar')
    im.rbepwt.region_collection_at_level[1][0].show(True,px_value=False,path_color='green',border_thickness=0.02)
    im.rbepwt.region_collection_at_level[2][0].show(False,px_value=False,path_color='green',border_thickness=0.02)
    print(im.rbepwt.region_collection_at_level[1].points)
    print(im.rbepwt.region_collection_at_level[2].points)

#generate_epwt()
generate_easypath()
#generate_gradpath()

