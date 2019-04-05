import matplotlib.pyplot as plt
from sl_rm import load_result
from configs import *
import os

'''
rename the files in the baseline_compare
for _,_,files in os.walk('.'):
	for file in files:
		paras = file.split('decision_function_shape')
		new_file = paras[0]+'decision-function-shape'+paras[1]
		os.rename(file,new_file)
'''

if __name__ == "__main__":
	labels = dict()
	contents = dict()
	os.chdir('results')
	for _, dirs, _ in os.walk("./", topdown=True):
		for method in dirs:
			if method != GA:
				labels.update({method: []})
				contents.update({method: []})
				for _, _, files in os.walk(method):
					for file in files:
						paras = file[0:-4].split('_')
						labels[method].append({})
						for para in paras:
							para = para.strip('()')
							key, value = para.split(', ')
							key = key.strip("'")
							value = value.strip("'")
							labels[method][-1].update({key: value})
						content = load_result(model_name=method,file_name=file)
						contents[method].append(content)
	print(labels)
	print(contents)

	
	# plot for ovo/ovr score
	plt.figure(figsize=(6, 6))
	plt.subplots_adjust(bottom=.05, top=0.97, hspace=.25, wspace=.15,
						left=.05, right=.99)
	scs_ovo = {x:[] for x in KERNELS}
	scs_ovr = {x:[] for x in KERNELS}
	f1s_ovo = {x:[] for x in KERNELS}
	f1s_ovr = {x:[] for x in KERNELS}
	for idx, paras in enumerate(labels[COMPARE]):
		kernel = paras['kernel']
		sc = contents[COMPARE][idx]['score']
		f1 = contents[COMPARE][idx]['f1_score']
		if paras['decision-function-shape']=='ovo':
			scs_ovo[kernel].append(sc)
			f1s_ovo[kernel].append(f1)
		elif paras['decision-function-shape']=='ovr':
			scs_ovr[kernel].append(sc)
			f1s_ovr[kernel].append(f1)
	for kernel in KERNELS:
		color = COLORS[KERNELS_MAP[kernel]]
		plt.subplot(2,2,1)
		plt.plot(CS,scs_ovo[kernel],color=color,label=kernel)
		plt.title('ovo score')
		plt.ylabel("score")
		plt.xlabel("C")
		plt.yticks(np.linspace(0,1,11))
		plt.legend(fontsize='xx-large')
		plt.subplot(2,2,2)
		plt.plot(CS,scs_ovr[kernel],color=color,label=kernel)
		plt.title('ovr score')
		plt.ylabel("score")
		plt.xlabel("C")
		plt.yticks(np.linspace(0,1,11))
		plt.legend(fontsize='xx-large')
		plt.subplot(2,2,3)
		plt.plot(CS,f1s_ovo[kernel],color=color,label=kernel)
		plt.title('ovo f1-score')
		plt.ylabel("f1-score")
		plt.xlabel("C")
		plt.yticks(np.linspace(0,1,11))
		plt.legend(fontsize='xx-large')
		plt.subplot(2,2,4)
		plt.plot(CS,f1s_ovr[kernel],color=color,label=kernel)
		plt.title('ovr f1-score')
		plt.ylabel("f1-score")
		plt.xlabel("C")
		plt.yticks(np.linspace(0,1,11))
		plt.legend(fontsize='xx-large')
	plt.show()


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