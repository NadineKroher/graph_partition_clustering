import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  pairwise_distances
from sklearn.cluster import spectral_clustering
import main
import os

# load data
data_file = './data/artificial.txt'
k = 2
data = np.loadtxt(data_file)
data = data[np.where(data[:, 2] <= k)[0], :]

# plot init
colors = [(0.2, 0.5, 0.9), 'k', (0.9, 0.7, 0.1)]
markers = ['o', '>', 's']

# affinity matrix
D = pairwise_distances(data[:, :2], data[:, :2])
D = D / np.max(D)
s = 1 - D
s = np.triu(s)

# proposed algorithm
q, k_clustering = main.getclusters(s, k)
proposed_labels = np.zeros((data.shape[0],))
for i in range(k):
    proposed_labels[np.asarray(list(k_clustering[i]))] = i

proposed_instance_colors = [colors[int(l)] for l in list(proposed_labels)]
proposed_instance_markers = [markers[int(l)] for l in list(proposed_labels)]

# normalized cut
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.spectral_clustering.html
normcut_labels = spectral_clustering(s, n_clusters=k)
normcut_instance_colors = [colors[int(l)] for l in list(normcut_labels)]
normcut_instance_markers = [markers[int(l)] for l in list(normcut_labels)]

# plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
for i in range(data.shape[0]):
    plt.scatter(data[i, 0],
                data[i, 1],
                edgecolors=proposed_instance_colors[i],
                facecolors='none',
                marker=proposed_instance_markers[i],
                s=15)
plt.title('proposed algorithm')

plt.subplot(1, 2, 2)
for i in range(data.shape[0]):
    plt.scatter(data[i, 0],
                data[i, 1],
                edgecolors=normcut_instance_colors[i],
                facecolors='none',
                marker=normcut_instance_markers[i],
                s=15)
plt.title('normalized cut')
plt.tight_layout()
plt.show()

#plt.savefig('./plots/' + os.path.basename(data_file)[:-3] + 'png')
