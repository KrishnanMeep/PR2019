import numpy as np
import matplotlib.pyplot as plt
from One import PCA, LDA

def PlotPCA( Y, C1, C2 ):
	plt.plot(Y[:50, C1], Y[:50, C2], 'C0.', label = "Setosa")
	plt.plot(Y[50:100, C1], Y[50:100, C2], 'C1.', label = "Versicolor")
	plt.plot(Y[100:150, C1], Y[100:150, C2], 'C2.', label = "Viriginica")
	plt.xlabel("PCA Component " + str(C1+1))
	plt.ylabel("PCA Component " + str(C2+1))
	plt.legend()
	plt.show()

def PlotLDA( Y1, Y2, l1, l2 ):
	plt.gca().yaxis.set_major_locator(plt.NullLocator())
	plt.plot(Y1, len(Y1)*[1], 'C1.', label = l1)
	plt.plot(Y2, len(Y2)*[1],'C0.', label = l2)
	plt.xlabel("LDA Axis")
	plt.legend()
	plt.show()

if __name__ == '__main__':

	allofthem = []

	reader = open("Iris.data", "r")
	lines = reader.read().split("\n")
	for line in lines:
		line = line.split(",")
		allofthem.append(list(map(float, line[:-1])))

	allofthem = np.array(allofthem)

	Y, Eval, Evec = PCA( allofthem )
	print(Eval)
	print(Evec)

	PlotPCA( Y, 0, 1)
	PlotPCA( Y, 0, 2)
	PlotPCA( Y, 1, 2)

	Y1, Y2 = LDA(allofthem[0:50],allofthem[50:100])
	PlotLDA( Y1, Y2, "Setosa", "Versicolor" )

	Y1, Y2 = LDA(allofthem[0:50],allofthem[100:150])
	PlotLDA( Y1, Y2, "Setosa", "Viriginica" )

	Y1, Y2 = LDA(allofthem[50:100],allofthem[100:150])
	PlotLDA( Y1, Y2, "Versicolor", "Viriginica" )