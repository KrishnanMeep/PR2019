from One import UVDF, MVDF, UVGaussianSample, MVGaussianSample
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

samplesClass1 = []
samplesClass2 = []
samplesClass3 = []

#############################################################################################
#The next four Plot functions are for showing how the distributions
#of each class are like, assuming they are Gaussian
def PlotDUV():
	mu1 = np.array(sum(samplesClass1[:, 0])/len(samplesClass1))
	mu2 = np.array(sum(samplesClass2[:, 0])/len(samplesClass2))
	var1 = np.cov(samplesClass1[:, 0])
	var2 = np.cov(samplesClass2[:, 0])
	print("First Univariate Dichotomizer")
	print("Mean and Variance of w1: ", mu1, ", ", var1)
	print("Mean and Variance of w2: ", mu2, ", ", var2 )
	S = [ UVGaussianSample(mu1, var1) for i in range(200) ]
	R = [ UVGaussianSample(mu2, var2) for i in range(200) ]
	plt.hist(R)
	plt.hist(S)
	plt.show()

def PlotDMVI():
	mu1 = np.array(sum(samplesClass1[:, 0:2])/len(samplesClass1))
	mu2 = np.array(sum(samplesClass2[:, 0:2])/len(samplesClass2))
	var1 = np.cov(samplesClass1[:, 0:2].T)
	var2 = np.cov(samplesClass2[:, 0:2].T)
	print("First Multivariate Dichotomizer")
	print("Mean and Variance of w1: ", mu1)
	print(var1)
	print("Mean and Variance of w2: ", mu2, ", ")
	print(var2)
	S = np.array([ MVGaussianSample(mu1, var1, 2) for i in range(200) ])
	R = np.array([ MVGaussianSample(mu2, var2, 2) for i in range(200) ])
	plt.plot(R[:,0], R[:,1], '.')
	plt.plot(S[:,0], S[:,1], '.')
	plt.show()

def PlotDMVII():
	mu1 = np.array(sum(samplesClass1)/len(samplesClass1))
	mu2 = np.array(sum(samplesClass2)/len(samplesClass2))
	var1 = np.cov(samplesClass1.T)
	var2 = np.cov(samplesClass2.T)
	print("Second Multivariate Dichotomizer")
	print("Mean and Variance of w1: ", mu1 )
	print(var1)
	print("Mean and Variance of w2: ", mu2 )
	print(var2)
	S = np.array([ MVGaussianSample(mu1, var1, 3) for i in range(200) ])
	R = np.array([ MVGaussianSample(mu2, var2, 3) for i in range(200) ])
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(R[:,0], R[:,1], R[:, 2])
	ax.scatter(S[:,0], S[:,1], S[:, 2])
	plt.show()

def PlotDMVIII():
	mu1 = np.array(sum(samplesClass1)/len(samplesClass1))
	mu2 = np.array(sum(samplesClass2)/len(samplesClass2))
	mu3 = np.array(sum(samplesClass3)/len(samplesClass3))
	var1 = np.cov(samplesClass1.T)
	var2 = np.cov(samplesClass2.T)
	var3 = np.cov(samplesClass3.T)
	print("Classifier for all three classes")
	print("Mean and Variance of w1: ", mu1)
	print(var1)
	print("\nMean and Variance of w2: ", mu2)
	print(var2)
	print("\nMean and Variance of w3: ", mu3)
	print(var3)
	S = np.array([ MVGaussianSample(mu1, var1, 3) for i in range(200) ])
	R = np.array([ MVGaussianSample(mu2, var2, 3) for i in range(200) ])
	T = np.array([ MVGaussianSample(mu3, var3, 3) for i in range(200) ])
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(R[:,0], R[:,1], R[:, 2])
	ax.scatter(S[:,0], S[:,1], S[:, 2])
	ax.scatter(T[:,0], T[:,1], T[:, 2])
	plt.show()


##############################################################################################
#First two classes, first feature x1
def FirstUVDichotomizer( x ):
	Pw1 = Pw2 = 0.5
	mu1 = np.array(sum(samplesClass1[:, 0])/len(samplesClass1))
	mu2 = np.array(sum(samplesClass2[:, 0])/len(samplesClass2))
	var1 = np.cov(samplesClass1[:, 0])
	var2 = np.cov(samplesClass2[:, 0])

	DichoResult = UVDF(x, mu1, var1, Pw1) - UVDF(x, mu2, var2, Pw2)
	return DichoResult

#First two classes, first two features x1 & x2
def FirstMVDichotomizer( x ):
	Pw1 = Pw2 = 0.5
	mu1 = np.array(sum(samplesClass1[:, 0:2])/len(samplesClass1))
	mu2 = np.array(sum(samplesClass2[:, 0:2])/len(samplesClass2))
	var1 = np.cov(samplesClass1[:, 0:2].T)
	var2 = np.cov(samplesClass2[:, 0:2].T)
	DichoResult = MVDF(x, mu1, var1, Pw1) - MVDF(x, mu2, var2, Pw2)
	return DichoResult

#First two classes, all features x1, x2 & x3
def SecondMVDichotomizer( x ):
	Pw1 = Pw2 = 0.5
	mu1 = np.array(sum(samplesClass1)/len(samplesClass1))
	mu2 = np.array(sum(samplesClass2)/len(samplesClass2))
	var1 = np.cov(samplesClass1.T)
	var2 = np.cov(samplesClass2.T)

	DichoResult = MVDF(x, mu1, var1, Pw1) - MVDF(x, mu2, var2, Pw2)
	return DichoResult

#All three classes, all three features
def ThirdMVDF( x, Pw1, Pw2, Pw3 ):
	mu1 = np.array(sum(samplesClass1)/len(samplesClass1))
	mu2 = np.array(sum(samplesClass2)/len(samplesClass2))
	mu3 = np.array(sum(samplesClass3)/len(samplesClass3))
	var1 = np.cov(samplesClass1.T)
	var2 = np.cov(samplesClass2.T)
	var3 = np.cov(samplesClass3.T)

	results = []
	results.append((MVDF(x, mu1, var1, Pw1), "w1"))
	results.append((MVDF(x, mu2, var2, Pw2), "w2"))
	results.append((MVDF(x, mu3, var3, Pw3), "w3"))

	A = max(results)
	print(x, " belongs to class ", A[1])

##############################################################################################

if __name__ == '__main__':
	reader = open("DHS.data", "r")
	lines = reader.read().split("\n")

	for line in lines:
		line = line.split(",")
		if line[-1].strip() == "w1":
			samplesClass1.append( list(map(float, line[:-1])))
		if line[-1].strip() == "w2":
			samplesClass2.append( list(map(float, line[:-1])))
		if line[-1].strip() == "w3":
			samplesClass3.append( list(map(float, line[:-1])))

	samplesClass1 = np.array(samplesClass1)
	samplesClass2 = np.array(samplesClass2)
	samplesClass3 = np.array(samplesClass3)

#################################################
#To view the mean and covariance of the data and the distribution, uncomment the following
#
	#PlotDUV()
	#PlotDMVI()
	#PlotDMVII()
	#PlotDMVIII()
#
#################################################
	'''
	x = np.array(-5.5)
	ret = FirstUVDichotomizer(x)
	if ret > 0:	print(x, " belongs to class w1")
	else:	print(x, "belongs to class w2")

	x = [0,1.6]
	ret = FirstMVDichotomizer(x)
	if ret > 0:	print(x, " belongs to class w1")
	else:	print(x, "belongs to class w2")

	x = [-1,3.4,1]
	ret = SecondMVDichotomizer(x)
	if ret > 0:	print(x, " belongs to class w1")
	else:	print(x, "belongs to class w2")


	#Finding empirical training error for Dichotomizers
	countUV = countMV1 = countMV2 = 0
	for i in range(0,10):
		ret = FirstUVDichotomizer(samplesClass1[i, 0])
		if ret < 0 : countUV +=1
		ret = FirstUVDichotomizer(samplesClass2[i, 0])
		if ret >= 0: countUV +=1

		ret = FirstMVDichotomizer(samplesClass1[i, 0:2])
		if ret < 0 : countMV1 +=1
		ret = FirstMVDichotomizer(samplesClass2[i, 0:2])
		if ret >= 0: countMV1 +=1

		ret = SecondMVDichotomizer(samplesClass1[i])
		if ret < 0 : countMV2 +=1
		ret = SecondMVDichotomizer(samplesClass2[i])
		if ret >= 0: countMV2 +=1

	print("Empirical Training Error for Univariate (One Feature) Dichotomizer: ", countUV/20*100, "%")
	print("Empirical Training Error for Multivariate (Two Features) Dichotomizer: ", countMV1/20*100, "%")
	print("Empirical Training Error for Multivariate (All Features) Dichotomizer: ", countMV2/20*100, "%")
	

	#Equal priors
	print("\nClassifying given four points with priors (0.33, 0.33, 0.33)")
	ThirdMVDF([1,2,1], 0.33, 0.33, 0.33)
	ThirdMVDF([5,3,2], 0.33, 0.33, 0.33)
	ThirdMVDF([0,0,0], 0.33, 0.33, 0.33)
	ThirdMVDF([1,0,0], 0.33, 0.33, 0.33)
	'''
	
	#Class1 has the highest prior
	print("Classifying given four points with priors (0.8, 0.1, 0.1)")
	ThirdMVDF([1,2,1], 0.8, 0.1, 0.1)
	ThirdMVDF([5,3,2], 0.8, 0.1, 0.1)
	ThirdMVDF([0,0,0], 0.8, 0.1, 0.1)
	ThirdMVDF([1,0,0], 0.8, 0.1, 0.1)
	