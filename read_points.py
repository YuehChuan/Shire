# -*- coding: utf-8 -*-
import numpy as np

list_x = []
list_y = []
list_intensitive = []
list_data = []

"""
Load data laser 721 points
"""
with open("2d_pointcloud.txt", mode="r", encoding="utf-8") as f:
    next(f)
    for line in f:
        list_x.append( float(line.split()[0]) )
        list_y.append( float(line.split()[1]) )
        list_intensitive.append( float(line.split()[3]) )
        list_data.append( (float(line.split()[0]), float(line.split()[1]), float(line.split()[2]) ) )


        #print(line.split()[0])

x = np.array(list_x)
y = np.array(list_y)
I = np.array(list_intensitive)
data = np.array(list_data)


"""
DBSCAN
"""
from sklearn.cluster import DBSCAN


db = DBSCAN(eps=0.5, min_samples=1).fit(data)
labels = db.labels_

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

from collections import Counter
Counter(labels)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=8)

    xy = data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

plt.title('Estimated number of clusters: %d' % n_clusters_)

plt.xlim(-10,10)
plt.ylim(-10,10)

plt.show()


"""
plot RAW DATA
"""
xmin=min(x)
xmax=max(x)
ymin=min(y)
ymax=max(y)
Imin=min(I)
Imax=max(I)

cm = plt.cm.get_cmap('Greens')

plt.xlim(-10,10)
plt.ylim(-10,10)
plt.scatter(x, y, c=I, vmin=Imin, vmax=Imax, cmap=cm, alpha=0.5)
plt.scatter(x, y, c=I, cmap=cm)
plt.show()






