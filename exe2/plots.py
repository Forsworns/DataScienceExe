import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import cosine_similarity
from load_data import load_data
from pre_process import pre_process
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def curse():
	fig = plt.figure()
	e = 0.01
	x = [100*i for i in  range(1,21)]
	y = [1-(1-e)**(10*i) for i in range(1,21)]
	# ax = plt.gca()
	ax = fig.add_subplot(1,1,1)
	ax.yaxis.grid(color='gray', linestyle='-', linewidth=1,alpha=0.3)
	ax.plot(x,y)
	plt.show()



def normalize(X):
	pass


def compute_dist(X):
	X = X[::100]
	similarities = cosine_similarity(X)
	fig = plt.figure()
	ax = Axes3D(fig)
	xx = []
	yy = []
	zz = []
	num = 0
	for i in range(0, similarities.shape[0]):
		for j in range(0, similarities.shape[1]):
			xx.append(i)
			yy.append(j)
			zz.append(similarities[i,j])
			if similarities[i,j]>-0.2 and similarities[i,j]<0.4:
				num = num + 1
	ax.scatter(xx,yy,zz)
	print(num/similarities.shape[0]/similarities.shape[1])
	ax.legend(loc='best')
	plt.show()

if __name__ == "__main__":
	curse()
	X, y = load_data()
	X_train, X_test, y_train, y_test = pre_process(X, y)

	lda = LinearDiscriminantAnalysis(n_components=50)
	lda.fit(X_train, y_train)
	X_train = lda.transform(X_train)
	X_test = lda.transform(X_test)
	
	compute_dist(X_train)
