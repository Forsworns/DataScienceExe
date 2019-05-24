from baseline import SVM_recommend, SVM_recommend_run, KNN_recommend_run
from sklearn.decomposition import PCA
from load_data import load_data
from pre_process import pre_process

if __name__ == "__main__":
	for data in range(3):
		X_src, y_src, X_tgt, y_tgt = load_data(data)
		pca = PCA(n_components=500)
		X_src = pca.fit_transform(X_src)
		X_tgt = pca.fit_transform(X_tgt)
		# baseline, train on the srouce domain, test on the target domain
		SVM_recommend_run("BASELINE_PCA", X_src, X_tgt, y_src, y_tgt,paras={"data":data})