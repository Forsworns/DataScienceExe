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
