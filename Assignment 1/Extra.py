import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from One import MVDF, MVGaussianSample, EuclideanM, Mahalanobis

#https://docs.google.com/forms/d/e/1FAIpQLSfpOOvuaAZFEho1GQ0pOzGzFdyRfOsD7CF3aScIFMWz-dhwRg/viewform?usp=form_confirm&edit2=2_ABaOnucjxVJCDICCwApm1iqcqMui7SPit4I67-OgCdtr

x = np.array([0.3,0.5,0.4])
mean = np.array([0.1,0.4,0.8])
covariance = np.array([ [3, 0, 0], [0, 3, 0], [0, 0, 3]])
prior = 1/4

print("Input vector x :", x)
print("Mean : ", mean)
print("Covariance :")
print(covariance)
'''
print("Prior : ", prior )
print("\nResult of DF on passing x : ", MVDF(x,mean,covariance,prior))


mu = np.array([3, 1])
mu2 = np.array([-1,-3])
sigma = np.array([[1,0],[0,1]])
sigma2 = np.array([[-4,0],[0,2]])
S = np.array([ MVGaussianSample(mu, sigma,2) for i in range(300) ])
R = np.array([ MVGaussianSample(mu2,sigma2,2) for i in range(300)])
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.plot( S[:, 0], S[:, 1], '.')
plt.plot( R[:, 0], R[:, 1], 'r.')
plt.show()

print("Distance between (3,0) and (0,3) is ", end = "")
print(EuclideanM(np.array([3,0]), np.array([0,3])))
'''
print("Distance between x and the mean : ", Mahalanobis(x,mean,covariance))
