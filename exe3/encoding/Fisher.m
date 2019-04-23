clear;
clc;
addpath(genpath('..'))
cd('..')
run('vl_setup')

data_path = '.\results\siftLD\siftLD.mat';
class_path = '.\results\siftLD\classes.mat';
numLD_path = '.\results\siftLD\numLD.mat';
fisher_dir = '.\results\fisher\';
if ~exist(fisher_dir,'dir')
    mkdir(fisher_dir)
end

load(data_path,'lds');
load(class_path,'file_nums');
numLabel = length(file_nums);
load(numLD_path,'numLD'); % the local descriptor number in a figure

%% train a GMM
tic;
numClusters = 100 ;
[means, covariances, priors] = vl_gmm(data, numClusters);
model_time = toc;

%% ecoding by GMM (encoding length is 4*numclusters?)
tic;
k = 1;
cursor = 1;
for i=1:numLabel
    for j=1:file_nums(i) 
        fig = lds(:,cursor:cursor+numLD(k)-1);
        cursor = cursor+numLD(k);
        
        encoding = vl_fisher(fig, means, covariances, priors);
        save(sprintf('%s%d%d.mat',fisher_dir,i,j),'encoding');
        k = k+1;
    end
end
encoding_time = toc;