import sys
from math import log, pi, sqrt, cos, exp
import numpy as np
import matplotlib.pyplot as plt

def EuclideanU( x, y ):
	return sqrt((x-y)**2)

def EuclideanM( x, y ):
	return sqrt(sum((x-y)**2))

def Mahalanobis( x, mean, covariance ):
	covInv = np.linalg.pinv(covariance)
	return sqrt(np.matmul(np.matmul( (x-mean).T, covInv), (x-mean)))


#Calculates probability of an input vector occuring in a distribution
def UVGaussianValue(x, mean, variance):
	A = 1/(2*pi)**2
	B = 1/variance
	C = exp(-0.5*(x-mean)**2/(variance)**2)
	return A*B*C

def MVGaussianValue(x, mean, covariance):
	covInv = np.linalg.pinv(covariance)
	covDet = np.linalg.det(covariance)
	A = 1/(2*pi)**(len(mean)/2)
	B = 1/(covDet)**2
	C = exp(-0.5*np.matmul(np.matmul((x-mean).T, covInv),(x-mean)))
	return A*B*C

#Uses Box Mueller Transform
def UVGaussianSample( mean, variance ):
	u1 = np.random.random()
	u2 = np.random.random()
	z0 = sqrt(-2*log(u1)) * cos(2*pi*u2) #Or sin
	return z0*variance + mean

def MVGaussianSample( mean, covariance, d ):
	Z = []
	for i in range(d):
		u1 = np.random.random()
		u2 = np.random.random()
		Z.append(sqrt(-2*log(u1)) * cos(2*pi*u2))	#Or sin
	Z = np.matmul(covariance, Z) + mean
	return Z

#Equation has been split into the distance and non distance parts for readability
def UVDF( x, mean, variance, prior ):
	A = -0.5*EuclideanU(x, mean)/variance
	B = -0.5*log(2*pi) -0.5*log(variance) + log(prior)
	return A+B

def MVDF( x, mean, covariance, prior ):
	d = len(x)
	covInv = np.linalg.pinv(covariance)
	covDet = np.linalg.det(covariance)

	A = -0.5*(Mahalanobis(x, mean, covariance)**2)
	B = -0.5*d*log(2*pi) -0.5*log(covDet) + log(prior)
	return A+B

