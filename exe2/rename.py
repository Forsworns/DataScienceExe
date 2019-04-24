import os


# rename the files in the baseline_compare
for _,_,files in os.walk('.'):
	for file in files:
		if 'minkowski' in file:
			paras = file.split('minkowski')
			print(file.split("_('p',"))
			p = file.split("_('p',")[1][1]
			new_file = paras[0]+'minkowski'+p+"').txt"
			os.rename(file,new_file)




