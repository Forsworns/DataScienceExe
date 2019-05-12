function X_src,y_src,X_tgt,y_tgt = load_data(type)
	if typt == 'a':
		src = csvread('data/a/A_A.csv');
		tgt = csvread('data/a/A_R.csv');
	end
	else if type == 'b':
		src = csvread('data/b/C_C.csv');
		tgt = csvread('data/b/C_R.csv');
	end
	else:
		src = csvread('data/c/P_P.csv');
		tgt = csvread('data/c/P_R.csv');
	end
	X_src = src(:,1:end-1);
	y_src = src(:,end);
	X_tgt = tgt(:,1:end-1);
	y_tgt = tgt(:,end);
end