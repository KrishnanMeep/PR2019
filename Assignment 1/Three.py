import numpy as np
import copy
from math import log
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from One import MVGaussianSample

setosa = []
versicolor = []
virginica = []

class LDFI:
	def __init__(self, X):
		self.computeMuSigma(X)

	def computeMuSigma(self, X):
		self.mean = np.array(sum(X)/len(X))
		self.cov = np.cov(X.T)
		self.var = 1

	def changeVar(self, var):
		self.var = var

	def Result(self, x):
		A = np.matmul(self.mean.T, x)/(self.var)**2
		B = -0.5*np.matmul(self.mean.T, self.mean)/(self.var)**2 + np.log(0.33)
		return A+B

class LDF:
	def __init__(self, X):
		self.computeMuSigma(X)

	def computeMuSigma(self, X):
		self.mean = np.array(sum(X)/len(X))
		self.cov = np.cov(X.T)
		self.covInv = np.linalg.pinv(self.cov)
		self.covDet = np.linalg.det(self.cov)

	def changeCov(self, cov):
		self.cov = cov

	def Result(self, x):
		A = np.matmul(np.matmul(self.covInv, self.mean).T, x)
		B = -0.5*np.matmul(np.matmul(self.mean.T, self.covInv), self.mean) + log(0.33)
		return A+B

class QDF:
	def __init__(self, X):
		self.computeMuSigma(X)

	def computeMuSigma(self, X):
		self.mean = np.array(sum(X)/len(X))
		self.cov = np.cov(X.T)
		self.covInv = np.linalg.pinv(self.cov)
		self.covDet = np.linalg.det(self.cov)

	def Result(self, x):
		A = -0.5*np.matmul(np.matmul(x.T, self.covInv), x)
		B = np.matmul(np.matmul(self.covInv, self.mean), x)
		C = -0.5*np.matmul(np.matmul(self.mean.T, self.covInv), self.mean) + log(0.33) -0.5*log(self.covDet)
		return A + B + C

if __name__ == '__main__':

	reader = open("Iris.data", "r")
	lines = reader.read().split("\n")
	for line in lines:
		line = line.split(",")
		if line[-1] == "Iris-setosa":
			setosa.append(list(map(float, line[:-1])))
		if line[-1] == "Iris-versicolor":
			versicolor.append(list(map(float, line[:-1])))
		if line[-1] == "Iris-virginica":
			virginica.append(list(map(float, line[:-1])))

	setosa = np.array(setosa)
	versicolor = np.array(versicolor)
	virginica = np.array(virginica)

	#Linear Discriminant Functions for each class (CASE II)
	SetosaLDF = LDF(setosa)
	VersicLDF = LDF(versicolor)
	VirginLDF = LDF(virginica)

	#Linear Discriminant Functions for each class (CASE I)
	SetosaLDFI = LDFI(setosa)
	VersicLDFI = LDFI(versicolor)
	VirginLDFI = LDFI(virginica)

	#Computing average covariance of three classes
	avgCov = (SetosaLDF.cov + VersicLDF.cov + VirginLDF.cov)/3
	for i in range(0,4):
		for j in range(0,4):
			avgCov[j][i] = (avgCov[i][j]+avgCov[j][i])/2
	SetosaLDF.changeCov(avgCov)
	VersicLDF.changeCov(avgCov)
	VirginLDF.changeCov(avgCov)

	#Computing average variance of the average covariance for case I
	avgCov2 = avgCov * np.eye(4,4)
	var = (np.mean(avgCov2[[0,1,2,3],[0,1,2,3]]))
	SetosaLDFI.changeVar(var)
	VersicLDFI.changeVar(var)
	VirginLDFI.changeVar(var)

	#Quadratic Discriminant Functions for each class
	SetosaQDF = QDF(setosa)
	VersicQDF = QDF(versicolor)
	VirginQDF = QDF(virginica)

###################################################################################

	#Count of misclassifications for each classifier
	countLDF = countLDFI = countQDF = 0

	#Calculating ETE for QDF based classifier
	for i in range(0, 50):
		A = SetosaQDF.Result(virginica[i])
		B = VersicQDF.Result(virginica[i])
		C = VirginQDF.Result(virginica[i])
		if C < A or C < B :
			countQDF += 1

		A = SetosaQDF.Result(setosa[i])
		B = VersicQDF.Result(setosa[i])
		C = VirginQDF.Result(setosa[i])
		if A < B or A < C :
			countQDF += 1

		A = SetosaQDF.Result(versicolor[i])
		B = VersicQDF.Result(versicolor[i])
		C = VirginQDF.Result(versicolor[i])
		if B < A or B < C :
			countQDF += 1

	#Calculating ETE for LDFI based classifier
	for i in range(0, 50):
		A = SetosaLDFI.Result(virginica[i])
		B = VersicLDFI.Result(virginica[i])
		C = VirginLDFI.Result(virginica[i])
		if C < A or C < B :
			countLDFI += 1

		A = SetosaLDFI.Result(setosa[i])
		B = VersicLDFI.Result(setosa[i])
		C = VirginLDFI.Result(setosa[i])
		if A < B or A < C :
			countLDFI += 1

		A = SetosaLDFI.Result(versicolor[i])
		B = VersicLDFI.Result(versicolor[i])
		C = VirginLDFI.Result(versicolor[i])
		if B < A or B < C :
			countLDFI += 1

	#Calculating ETE for LDF based classifier
	for i in range(0, 50):
		A = SetosaLDF.Result(virginica[i])
		B = VersicLDF.Result(virginica[i])
		C = VirginLDF.Result(virginica[i])
		if C < A or C < B :
			countLDF += 1

		A = SetosaLDF.Result(setosa[i])
		B = VersicLDF.Result(setosa[i])
		C = VirginLDF.Result(setosa[i])
		if A < B or A < C :
			countLDF += 1

		A = SetosaLDF.Result(versicolor[i])
		B = VersicLDF.Result(versicolor[i])
		C = VirginLDF.Result(versicolor[i])
		if B < A or B < C :
			countLDF += 1

	print("For the Setosa Class,")
	print("Mean : ", SetosaQDF.mean)
	print("Covariance : ")
	print(SetosaQDF.cov, "\n")

	print("For the Versicolor Class,")
	print("Mean : ", VersicQDF.mean)
	print("Covariance : ")
	print(VersicQDF.cov, "\n")

	print("For the Virginica Class,")
	print("Mean : ", VirginQDF.mean)
	print("Covariance : ")
	print(VirginQDF.cov, "\n")

	print("Average Covariance for Case II was taken as")
	print(avgCov)

	print("\nAverage Variance for Case I was taken as")
	print(var)

	print("\nEmprical Training Error for LDF (Case I) : ", (countLDFI/150)*100, "%")
	print("Emprical Training Error for LDF (Case II) : ", (countLDF/150)*100, "%")
	print("Emprical Training Error for QDF : ", (countQDF/150)*100, "%")

	S = np.array([ MVGaussianSample(SetosaQDF.mean, SetosaQDF.cov, 4) for i in range(200) ])
	R = np.array([ MVGaussianSample(VersicQDF.mean, VersicQDF.cov, 4) for i in range(200) ])
	T = np.array([ MVGaussianSample(VirginQDF.mean, VirginQDF.cov, 4) for i in range(200) ])
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(R[:,0], R[:,1], R[:, 2])
	ax.scatter(S[:,0], S[:,1], S[:, 2])
	ax.scatter(T[:,0], T[:,1], T[:, 2])
	plt.show()
