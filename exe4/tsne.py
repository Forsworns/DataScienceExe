from load_data import load_data
from pre_process import pre_process
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 

if __name__ == "__main__":
	for data in range(3):
		X_src, y_src, X_tgt, y_tgt = load_data(data)
		X_src = TSNE(n_components=2).fit_transform(X_src)
		X_tgt = TSNE(n_components=2).fit_transform(X_src)
		plt.figure()
		plt.subplot(121)
		for i in range(X_src.shape[0]):
			plt.scatter(X_src[i,0],X_src[i,1],color=plt.cm.Set1(y_src[i] / 65.))
		plt.subplot(122)
		for i in range(X_tgt.shape[0]):
			plt.scatter(X_tgt[i,0],X_tgt[i,1],color=plt.cm.Set1(y_tgt[i] / 65.))
		plt.show()