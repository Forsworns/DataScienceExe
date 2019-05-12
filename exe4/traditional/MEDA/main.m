% ±ÿ–Î”√knn
types = ['a','b','c'];
Acc = zeros(3,1);
for i=1:3
    [Xs,Ys,Xt,Yt] = load_data(types(i));
    % meda
    options.d = 20;
    options.rho = 1.0;
    options.p = 10;
    options.lambda = 10.0;
    options.eta = 0.1;
    options.T = 10;
    [Acc(i),~,~,~] = MEDA(Xs,Ys,Xt,Yt,options);
    fprintf('type %d: %.2f accuracy \n\n', i, Acc(i) * 100);
end
if ~exist('results/MEDA/','dir')
    mkdir('results/MEDA/');
end
save('results/MEDA/MEDA.txt','Acc','-ascii');

