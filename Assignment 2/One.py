import numpy as np
import matplotlib.pyplot as plt

def PCA( D ):
	mu = np.mean(D, axis = 0)
	X = D - mu
	S = np.matmul(X.T,X)
	EigenValues, EigenVectors = np.linalg.eig(S)

	EigenValues = np.array(EigenValues)
	EigenVectors = np.array(EigenVectors)

	Y = np.dot(X, EigenVectors.T)

	return Y, EigenValues, EigenVectors

def LDA( D1, D2 ):
	mu1 = np.mean(D1, axis = 0)
	mu2 = np.mean(D2, axis = 0)

	S1 = np.matmul((D1-mu1).T, (D1-mu1))
	S2 = np.matmul((D2-mu2).T, (D2-mu2))

	SW = S1 + S2
	muD = np.atleast_2d(mu1-mu2)
	SB = np.matmul(muD.T, muD)

	SWInv = np.linalg.pinv(SW)
	A = np.matmul(SWInv, SB)
	EigenValues, _ = np.linalg.eig(A)

	V = np.matmul(SWInv, muD.T)

	Y1 = np.dot(D1, V)
	Y2 = np.dot(D2, V)
	return Y1, Y2