clear;
clc;
addpath('..')
run('vl_setup')

data_path = '.\results\siftLD\siftLD.mat';
class_path = '.\results\siftLD\classes.mat';
encodes_path = '\results\BOW\encodes.mat';
numLD_path = '.\results\siftLD\numLD.mat';
load(data_path,'lds');
load(class_path,'file_nums');
numLabel = length(file_nums);
load(numLD_path,'numLD'); % the local descriptor number in a figure

%% train a GMM
numClusters = 100 ;
[means, covariances, priors] = vl_gmm(data, numClusters);

%% ecoding by GMM (encoding length is 4*numclusters?)
k = 1;
cursor = 1;
for i=1:numLabel
    for j=1:file_nums(i) 
        fig = lds(:,cursor:cursor+numLD(k));
        cursor = cursor+numLD(k);
        
        encoding = vl_fisher(fig, means, covariances, priors);
        save(sprintf("%d%d.mat",i,j),encoding);
        k = k+1;
    end
end
