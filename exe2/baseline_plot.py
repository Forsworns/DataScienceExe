import matplotlib.pyplot as plt
from sl_rm import load_result
from configs import *
import os

'''
# rename the files in the baseline_compare
for _,_,files in os.walk('.'):
	for file in files:
		paras = file.split('decision_function_shape')
		new_file = paras[0]+'decision-function-shape'+paras[1]
		os.rename(file,new_file)
'''




if __name__ == "__main__":
	# rename the files in the baseline_compare
	os.chdir('results')
	for _,dirs,_ in os.walk('./'):
		print(dirs)
		for method in dirs:
			os.chdir(method)
			for _, _, files in os.walk('./'):
				for file in files:
					print(file)
					if 'n_neighbors' in file:
						paras = file.split('n_neighbors')
						new_file = paras[0]+'n-neighbors'+paras[1]
						os.rename(file,new_file)
			os.chdir('..')

	labels = dict()
	contents = dict()
	for _, dirs, _ in os.walk("./", topdown=True):
		for method in dirs:
			labels.update({method: []})
			contents.update({method: []})
			for _, _, files in os.walk(method):
				for file in files:
					paras = file[0:-4].split('_')
					labels[method].append({})
					print(paras)
					for para in paras:
						para = para.strip('()')
						if ',' in para:
							key, value = para.split(', ')
						else:
							key, value = para.split('， ')
						key = key.strip("'")
						value = value.strip("'")
						labels[method][-1].update({key: value})
					content = load_result(model_name=method,file_name=file)
					contents[method].append(content)
	print(labels)
	print(contents)

	compareLabel = labels[COMPARE]
	compareData = contents[COMPARE]
	plt.figure(figsize=(6, 6))
	ax = plt.gca()
	ax.yaxis.grid(color='gray', linestyle='-', linewidth=1,alpha=0.3)
	for dist in DIST_LIST:
		sc = [compareData[idx]['score'] for idx,item in enumerate(compareLabel) if item['m']==dist]
		color = COLORS[DIST_MAP[dist]]
		plt.plot(NEIGHBORS,sc,color=color,label=dist)
		plt.title('Baseline score on the original dataset',fontsize=20)
		plt.ylabel("score",fontsize=20)
		plt.xlabel("neighbor number",fontsize=20)
		plt.yticks(np.linspace(0,1,11))
		plt.legend(fontsize='xx-large')
	plt.show()

	'''
	# plot for baseline
	scs = []
	f1s = []
	for idx, paras in enumerate(labels[BASELINE]):
		sc = contents[BASELINE][idx]['score']
		f1 = contents[BASELINE][idx]['f1_score']
		scs.append(sc)
		f1s.append(f1)
	plt.figure(figsize=(3, 6))
	plt.subplot(1,2,1)
	plt.plot(CS,scs)
	plt.title('baseline score')
	plt.ylabel("score")
	plt.xlabel("C")
	plt.subplot(1,2,2)
	plt.plot(CS,f1s)
	plt.title('baseline f1-score')
	plt.ylabel("f1-score")
	plt.xlabel("C")
	plt.show()
	

	# plot for summary
	plt.figure(figsize=(3, 6))
	plt.subplot(1,2,1)
	plt.plot([i for i in range(2048)], [scs[0] for _ in range(2048)],color=COLORS[0],label=BASELINE)
	plt.subplot(1,2,2)
	plt.plot([i for i in range(2048)], [f1s[0] for _ in range(2048)],color=COLORS[0],label=BASELINE)
	# plot for AUC ROC
	zips = []
	for idx, paras in enumerate(labels[AUC]):
		fn = int(paras['feature-num'])
		sc = contents[AUC][idx]['score']
		f1 = contents[AUC][idx]['f1_score']
		zips.append((fn,sc,f1))
	zips.sort(key=lambda x:x[0])
	fns = [fn for fn,_,_ in zips] 
	scs = [sc for _,sc,_ in zips] 
	f1s = [f1 for _,_,f1 in zips] 
	plt.subplot(1,2,1)
	plt.plot(fns,scs,color=COLORS[1],label=AUC)
	plt.subplot(1,2,2)
	plt.plot(fns,f1s,color=COLORS[1],label=AUC)
	# plot for forward_univariable_feature
	zips = []
	for idx, paras in enumerate(labels[F_UF]):
		fn = int(paras['k-best'])
		sc = contents[F_UF][idx]['score']
		f1 = contents[F_UF][idx]['f1_score']
		zips.append((fn,sc,f1))
	zips.sort(key=lambda x:x[0])
	fns = [fn for fn,_,_ in zips] 
	scs = [sc for _,sc,_ in zips] 
	f1s = [f1 for _,_,f1 in zips] 
	plt.subplot(1,2,1)
	plt.plot(fns,scs,color=COLORS[2],label=F_UF)
	plt.subplot(1,2,2)
	plt.plot(fns,f1s,color=COLORS[2],label=F_UF)
	# plot for backward variance threshold
	scs = []
	f1s = []
	fns = []
	for idx, paras in enumerate(labels[B_VT]):
		fn = paras['feature-num']
		fns.append(fn)
		sc = contents[B_VT][idx]['score']
		f1 = contents[B_VT][idx]['f1_score']
		scs.append(sc)
		f1s.append(f1)
	plt.subplot(1,2,1)
	plt.plot(fns,scs,color=COLORS[3],label=B_VT)
	plt.subplot(1,2,2)
	plt.plot(fns,f1s,color=COLORS[3],label=B_VT)
	# plot for backward_select_from_model
	zips = []
	for idx, paras in enumerate(labels[B_SFM]):
		fn = int(paras['max-features'])
		sc = contents[B_SFM][idx]['score']
		f1 = contents[B_SFM][idx]['f1_score']
		zips.append((fn,sc,f1))
	zips.sort(key=lambda x:x[0])
	fns = [fn for fn,_,_ in zips] 
	scs = [sc for _,sc,_ in zips] 
	f1s = [f1 for _,_,f1 in zips] 
	plt.subplot(1,2,1)
	plt.plot(fns,scs,color=COLORS[4],label=B_SFM)
	plt.legend(fontsize='xx-large')
	plt.title("feature selection score")
	plt.subplot(1,2,2)
	plt.plot(fns,f1s,color=COLORS[4],label=B_SFM)
	plt.legend(fontsize='xx-large')
	plt.title("feature selection f1-score")
	plt.show()
	'''


	'''
	# plot for forward_univariable_feature
	zips = []
	for idx, paras in enumerate(labels[F_UF]):
		fn = int(paras['k-best'])
		sc = contents[F_UF][idx]['score']
		f1 = contents[F_UF][idx]['f1_score']
		zips.append((fn,sc,f1))
	zips.sort(key=lambda x:x[0])
	fns = [fn for fn,_,_ in zips] 
	scs = [sc for _,sc,_ in zips] 
	f1s = [f1 for _,_,f1 in zips] 
	plt.figure(figsize=(3, 6))
	plt.subplot(1,2,1)
	plt.plot(fns,scs)
	plt.title('forward univariable feature score')
	plt.ylabel("score")
	plt.xlabel("feature numbers")
	plt.subplot(1,2,2)
	plt.plot(fns,f1s)
	plt.title('forward univariable feature f1-score')
	plt.ylabel("f1-score")
	plt.xlabel("feature numbers")
	plt.show()


	# plot for backward_variance_threshold
	scs = []
	f1s = []
	fns = []
	for idx, paras in enumerate(labels[B_VT]):
		fn = paras['feature-num']
		fns.append(fn)
		sc = contents[B_VT][idx]['score']
		f1 = contents[B_VT][idx]['f1_score']
		scs.append(sc)
		f1s.append(f1)
	plt.figure(figsize=(3, 6))
	plt.subplot(1,2,1)
	plt.plot(fns,scs)
	plt.title('backward variance threshold score')
	plt.ylabel("score")
	plt.xlabel("feature numbers")
	plt.subplot(1,2,2)
	plt.plot(fns,f1s)
	plt.title('backward variance threshold f1-score')
	plt.ylabel("f1-score")
	plt.xlabel("feature numbers")
	plt.show()

	# plot for backward_select_from_model
	zips = []
	for idx, paras in enumerate(labels[B_SFM]):
		fn = int(paras['max-features'])
		sc = contents[B_SFM][idx]['score']
		f1 = contents[B_SFM][idx]['f1_score']
		zips.append((fn,sc,f1))
	zips.sort(key=lambda x:x[0])
	fns = [fn for fn,_,_ in zips] 
	scs = [sc for _,sc,_ in zips] 
	f1s = [f1 for _,_,f1 in zips] 
	plt.figure(figsize=(3, 6))
	plt.subplot(1,2,1)
	plt.plot(fns,scs)
	plt.title('backward select from model score')
	plt.ylabel("score")
	plt.xlabel("feature numbers")
	plt.subplot(1,2,2)
	plt.plot(fns,f1s)
	plt.title('backward select from model f1-score')
	plt.ylabel("f1-score")
	plt.xlabel("feature numbers")
	plt.show()
	'''