import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from One import PCA
from mpl_toolkits.mplot3d import Axes3D

def PlotComps( Y, L1, L2 ):
	plt.plot(Y[:,0], Y[:,1], 'C0.')
	plt.xlabel(L1)
	plt.ylabel(L2)
	plt.legend()
	plt.show()

if __name__ == '__main__':
	X, _ = make_swiss_roll(1000)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	#ax.scatter(X[:,0], X[:,1], X[:, 2], cmap = plt.get_cmap('Oranges'))
	#plt.show()

	T = KernelPCA(n_components = 2, kernel = "linear")
	YKpca = T.fit_transform(X)

	T = KernelPCA(n_components = 2, kernel = "rbf")
	YKpca2 = T.fit_transform(X)

	LLE = LocallyLinearEmbedding(n_components = 2, n_neighbors = 5)
	YLLE = LLE.fit_transform(X)

	LLE = LocallyLinearEmbedding(n_components = 2, n_neighbors = 150)
	YLLE2 = LLE.fit_transform(X)

	Y, _, _ = PCA(X)
	PlotComps( Y, "PCA Component 1", "PCA Component 2")
	PlotComps( YKpca, "kPCA Component 1", "kPCA Component 2")
	PlotComps( YKpca2, "kPCA Component 1", "kPCA Component 2")
	PlotComps( YLLE, "LLE Component 1", "LLE Component 2")
	PlotComps( YLLE2, "LLE Component 1", "LLE Component 2")

	#PUNCTURE META
	X = np.concatenate( (X[0:120], X[150:240], X[300:500], X[500:540], X[600:900], X[950:]) )
	ax.scatter(X[:,0], X[:,1], X[:, 2], cmap = plt.get_cmap('Oranges'))
	plt.show()

	T = KernelPCA(n_components = 2, kernel = "linear")
	YKpca = T.fit_transform(X)
	T = KernelPCA(n_components = 2, kernel = "rbf")
	YKpca2 = T.fit_transform(X)
	LLE = LocallyLinearEmbedding(n_components = 2, n_neighbors = 5)
	YLLE = LLE.fit_transform(X)
	Y, _, _ = PCA(X)
	PlotComps( Y, "PCA Component 1", "PCA Component 2")
	PlotComps( YKpca, "kPCA Component 1", "kPCA Component 2")
	PlotComps( YKpca2, "kPCA Component 1", "kPCA Component 2")
	PlotComps( YLLE, "LLE Component 1", "LLE Component 2")