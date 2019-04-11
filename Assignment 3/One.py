import numpy as np
import copy
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score, davies_bouldin_score, f1_score, confusion_matrix
from scipy.stats import mode

def Euclidean( x, y ):
	return sqrt(sum((x-y)**2))

def KMeans( D, k ):
	centroids = np.array(D[np.random.randint(0, len(D)-1, size = k)])
	iterations = 0
	clusters = np.array([-1]*len(D))
	prevClusters = copy.deepcopy(clusters)

	while iterations < 50:
		for i in range(len(D)):
			distances = [Euclidean(D[i], x) for x in centroids]
			clusters[i] = np.argmin(distances)

		if np.array_equal(clusters, prevClusters):
			break
		prevClusters = copy.deepcopy(clusters)
		centroids = np.array([ np.mean(D[np.where(clusters == x)], axis = 0) for x in range(k)])
		iterations += 1
	return clusters, centroids

def Measures( D, actualClusters, clusters, QC ):

	print("Internal Measures")
	print("Silhouette Coefficient : ", silhouette_score(D, clusters, metric="euclidean"))
	print("Davies Bouldin Score : ", davies_bouldin_score(D, clusters))
	
	if QC == 0 :
		print("No class labels present, so no external measures can be computed")
		return

	#Calculating purity!
	P = 0

	for x in set(actualClusters):
		A = actualClusters[actualClusters == x]
		B = clusters[clusters == x]
		P += min( (A == mode(A)[0][0]).sum(), (B == mode(B)[0][0]).sum())
	P = P/len(allofthem)

	print("\nExternal Measures")
	print("Purity: ", P)
	print("F Measure : ", f1_score(actualClusters, clusters, average = 'macro'))

	print("Confusion Matrix:")
	print(confusion_matrix(actualClusters, clusters))

if __name__ == '__main__':
	
	allofthem = []
	reader = open("Iris.data", "r")
	lines = reader.read().split("\n")
	for line in lines:
		line = line.split(",")
		allofthem.append(list(map(float, line[:-1])))
	allofthem = np.array(allofthem)
	
	'''
	#Initial 3D plot
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter( allofthem[:50, 0], allofthem[:50, 1], allofthem[:50, 2], c = 'C0', label = 'Setosa', alpha = 0.4)
	ax.scatter( allofthem[50:100, 0], allofthem[50:100, 1], allofthem[50:100, 2], c = 'C1', label = 'Versicolor', alpha = 0.4)
	ax.scatter( allofthem[100:150, 0], allofthem[100:150, 1], allofthem[100:150, 2], c = 'C2', label = 'Viriginica', alpha = 0.4)
	plt.legend()
	plt.show()
	'''
	
	k = 3
	clusters, centroids = KMeans( allofthem, k )

	actualClusters = np.array([0] * 50 + [1] * 50 + [2] * 50)
	Measures( allofthem, actualClusters, clusters, 1)
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter( centroids[:,0], centroids[:,1], centroids[:,2], c = 'C3' )

	for x,c in zip(range(k), ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']):
		A = allofthem[np.where(clusters == x)]
		ax.scatter( A[:, 0], A[:, 1], A[:, 2], c = c, label = 'Cluster '+ str(x), alpha = 0.4)
	plt.legend()
	plt.show()
	

	'''
	#For that other dataset
	
	allofthem = []
	reader = open("3D_spatial_network.txt", "r")
	lines = reader.read().split("\n")
	lines.pop()
	for line in lines:
		line = line.split(",")
		allofthem.append(list(map(float, line[1:])))
	
	allofthem = np.array(allofthem)

	k = 7
	clusters, centroids = KMeans( allofthem[:50000], k )
	Measures( allofthem[:50000], [], clusters, 0)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter( centroids[:,0], centroids[:,1], centroids[:,2], c = 'C3' )

	for x,c in zip(range(k), ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']):
		A = allofthem[np.where(clusters == x)]
		ax.scatter( A[:, 0], A[:, 1], A[:, 2], c = c, label = 'Cluster '+ str(x), alpha = 0.4)
	plt.legend()
	plt.show()
	'''
	