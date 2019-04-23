clear;
clc;
addpath(genpath('..'))
cd('..')
run('vl_setup')

data_path = '.\results\siftLD\siftLD.mat';
class_path = '.\results\siftLD\classes.mat';
numLD_path = '.\results\siftLD\numLD.mat';
vlad_dir = '.\results\vlad\';
if ~exist(vlad_dir,'dir')
    mkdir(vlad_dir)
end

load(data_path,'lds');
load(class_path,'file_nums');
numLabel = length(file_nums);
load(numLD_path,'numLD'); % the local descriptor number in a figure

%% the clusters of local desciptors
tic;
numClusters = 100;
centers = vl_kmeans(lds, numClusters);
kdtree = vl_kdtreebuild(centers) ;
model_time = toc;
%% encoding (the length of encoding is 2*clusters)
tic;
k = 1;
cursor = 1;
for i=1:numLabel
    for j=1:file_nums(i) 
        fig = lds(:,cursor:cursor+numLD(k)-1);
        cursor = cursor+numLD(k);
        
        nn = vl_kdtreequery(kdtree, centers, fig) ;
        assignments = zeros(numClusters,size(fig,2));
        % sub2ind() transforms sub-index to linear-index
        % nn is the nearest cluster neighbor, points should be assigned to this cluster
        assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
        
        encoding = vl_vlad(fig,centers,assignments);
        save(sprintf('%s%d_%d.mat',vlad_dir,i,j),'encoding');
        k = k+1;
    end
end
encoding_time = toc;