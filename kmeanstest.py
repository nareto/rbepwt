import sklearn.cluster as skluster
import numpy as np
import matplotlib.pyplot as plt

centers = [(0,0),(10,10), (0,10), (-5,-8)]
nclusters = len(centers)
points = {}
points_per_cluster = 20
colors = ['red','green','blue','black']

for label,center in enumerate(centers):
    var = 10
    #points[label] = np.random.normal(loc=center,scale=var,size=(2,points_per_cluster))
    points[label] = np.random.normal(loc=center,scale=var,size=(points_per_cluster,2))

matrix = None
for label,p in points.items():
    if matrix is not None:
        matrix = np.vstack((matrix,p))
    else:
        matrix = p

#plt.scatter(matrix[:,0],matrix[:,1])


#print(points)

km = skluster.KMeans(nclusters)
km.fit(matrix)
kmlabels = km.labels_

for idx,vec in enumerate(matrix):
    #plt.scatter(vec[0],vec[1],
    plt.plot(vec[0],vec[1],'.',color=colors[kmlabels[idx]])
    print(vec)

plt.show()
