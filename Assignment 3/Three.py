import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import SpectralClustering
from One import Measures, KMeans

if __name__ == '__main__':
	allofthem = []
	reader = open("3D_spatial_network.txt", "r")
	lines = reader.read().split("\n")
	lines.pop()
	for line in lines:
		line = line.split(",")
		allofthem.append(list(map(float, line[1:])))
	allofthem = np.array(allofthem)

	k = 4
	clusters = SpectralClustering(n_clusters = k, assign_labels = 'discretize', random_state = 0).fit(allofthem[:10000])
	clusters = clusters.labels_

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	#ax.scatter( centroids[:,0], centroids[:,1], centroids[:,2], c = 'C3' )

	for x,c in zip(range(k), ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']):
		A = allofthem[np.where(clusters == x)]
		ax.scatter( A[:, 0], A[:, 1], A[:, 2], c = c, label = 'Cluster '+ str(x), alpha = 0.4)
	plt.legend()
	plt.show()

	Measures( allofthem[:10000], [], clusters, 0 )
	k = 4
	clusters, centroids = KMeans( allofthem[:10000], k )
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	#ax.scatter( centroids[:,0], centroids[:,1], centroids[:,2], c = 'C3' )

	for x,c in zip(range(k), ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']):
		A = allofthem[np.where(clusters == x)]
		ax.scatter( A[:, 0], A[:, 1], A[:, 2], c = c, label = 'Cluster '+ str(x), alpha = 0.4)
	plt.legend()
	plt.show()
	Measures( allofthem[:10000], [], clusters, 0)

