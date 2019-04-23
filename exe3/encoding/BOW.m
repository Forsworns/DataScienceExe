clear;
clc;
addpath(genpath('..'))
cd('..')
run('vl_setup')

data_path = '.\results\siftLD\siftLD.mat';
class_path = '.\results\siftLD\classes.mat';
numLD_path = '.\results\siftLD\numLD.mat';
bow_dir = '.\results\bow\';
if ~exist(bow_dir,'dir')
    mkdir(bow_dir)
end

load(data_path,'lds');
load(class_path,'file_nums');
numLabel = length(file_nums);
load(numLD_path,'numLD'); % the local descriptor number in a figure

%% build word dictionary (center)
tic;
numClusters = 100; % encode to a word vector of length numClusters
centers = vl_kmeans(lds, numClusters);
kdtree = vl_kdtreebuild(centers) ;
model_time = toc;
%% encoding
tic;
k = 1;
cursor = 1;
for i=1:numLabel
    for j=1:file_nums(i) 
        fig = lds(:,cursor:cursor+numLD(k)-1);
        cursor = cursor+numLD(k);
        
        nn = vl_kdtreequery(kdtree, centers, fig);
        
        encoding = bag_of_words(nn,numClusters);
        save(sprintf('%s%d_%d.mat',bow_dir,i,j),'encoding');
        k = k+1;
    end
end
encoding_time = toc;

function encoded = bag_of_words(nn,numClusters)
    encoded = zeros(numClusters,1);
    for i =1:length(nn)
        encoded(nn(i)) = encoded(nn(i)) + 1;
    end
end