types = ['a','b','c'];
Acc = zeros(3,2);
subspace_dim_d = 50;

for i=1:3
    [Xs,Ys,Xt,Yt] = load_data(types(i));
    Xss = pca(Xs);
    Xtt = pca(Xt);
    Xss = Xss(:,1:subspace_dim_d);
    Xtt = Xtt(:,1:subspace_dim_d);
    [accuracy_sa_nn,accuracy_sa_svm] = Subspace_Alignment(Xs,Xt,Ys,Yt,Xss,Xtt);
    fprintf('NN SA Accuacry \t %1.2f \n',mean(accuracy_sa_nn));
    fprintf('SVM SA Accuacry \t %1.2f \n',mean(accuracy_sa_svm));
    Acc(i,:) = [accuracy_sa_nn,accuracy_sa_svm];
end

if ~exist('results/SA/','dir')
    mkdir('results/SA/');
end
save('results/SA/SA.txt','Acc','-ascii');