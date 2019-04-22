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

%% the clusters of local desciptors
numClusters = 100;
centers = vl_kmeans(data, numClusters);
kdtree = vl_kdtreebuild(centers) ;
%% encoding (the length of encoding is 2*clusters)
k = 1;
cursor = 1;
for i=1:numLabel
    for j=1:file_nums(i) 
        fig = lds(:,cursor:cursor+numLD(k));
        cursor = cursor+numLD(k);
        
        nn = vl_kdtreequery(kdtree, centers, fig) ;
        assignments = zeros(numClusters,100);
        % sub2ind() transforms sub-index to linear-index
        % nn is the nearest cluster neighbor, points should be assigned to this cluster
        assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
        
        encoding = vl_vlad(data,centers,assignments);
        save(sprintf("%d%d.mat",i,j),encoding);
        k = k+1;
    end
end