import rbepwt
import numpy as np

npoints = 40
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
        
region = rbepwt.Region(random_walk)
#region.show(title='Original')
region.pprint()

easypath = region.easy_path(level=None)
easypath.show(True,title='EasyPath')
easypath.pprint()
