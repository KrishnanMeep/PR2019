import numpy as np
import matplotlib.pyplot as plt
from One import PCA, LDA

def PlotPCA( Y, C1, C2 ):
	plt.plot(Y[:4, C1], Y[0:4, C2], 'C0.', label = "Healthy")
	plt.plot(Y[44:50, C1], Y[44:50, C2], 'C1.', label = "Has Cancer")
	plt.xlabel("PCA Component " + str(C1+1))
	plt.ylabel("PCA Component " + str(C2+1))
	plt.legend()
	plt.show()

def MakeScree( EVals ):
	plt.bar(np.arange(len(EVals)), EVals)
	plt.show()

def PlotLDA( Y1, Y2, l1, l2 ):
	plt.gca().yaxis.set_major_locator(plt.NullLocator())
	plt.plot(Y1, len(Y1)*[1], 'C1.', label = l1)
	plt.plot(Y2, len(Y2)*[1],'C0.', label = l2)
	plt.xlabel("LDA Axis")
	plt.legend()
	plt.show()

if __name__ == '__main__':

	#In the UCI Arcene training set, the first 44 are healthy and the rest are not healthy

	allofthem = []
	reader = open("arcene_train.data", "r")
	lines = reader.read().split("\n")
	lines.pop()
	print(len(lines))
	for line in lines:
		line = line.split()
		allofthem.append(list(map(float, line[:-1])))

	Y, EVals, EVecs = PCA( allofthem )

	PlotPCA( Y, 0, 1 )

	MakeScree( EVals )

	print(sum(EVals[:10])/sum(EVals)*100)
	print(sum(EVals[:19])/sum(EVals)*100)
	print(sum(EVals[:42])/sum(EVals)*100)
	print(sum(EVals[:84])/sum(EVals)*100)

	Y1, Y2 = LDA(allofthem[0:44], allofthem[44:])
	PlotLDA( Y1[:4], Y2[:5], "Healthy", "Has Cancer")