clear;
clc;
addpath(genpath('..'))
cd('..')
run('vl_setup')

data_path = '.\results\deep\feature_101.txt';
result_dir = '.\results\deep\';
data = importdata(data_path);
cell_size = 128;
cell_num = 100;
fig_num = size(data,1);
lds = zeros([cell_size,fig_num*cell_num]);
for f = 1:fig_num
    for i = 1:cell_num
        lds(:,(f-1)*cell_num+i) = data(f,(i-1)*cell_size+1:i*cell_size)';
    end
end

%% build word dictionary (center)
tic;
numClusters = 100; % encode to a word vector of length numClusters
centers = vl_kmeans(lds, numClusters);
kdtree = vl_kdtreebuild(centers) ;
model_time = toc;
%% encoding
tic;
features = zeros([fig_num,numClusters]);
for i=1:fig_num
    fig = lds(:,(i-1)*cell_num+1:i*cell_num);
    nn = vl_kdtreequery(kdtree, centers, fig);
    features(i,:) = bag_of_words(nn,numClusters);
end
save([result_dir,'deep_bow.txt'],'features','-ascii');
encoding_time = toc;

function encoded = bag_of_words(nn,numClusters)
    encoded = zeros(numClusters,1);
    for i =1:length(nn)
        encoded(nn(i)) = encoded(nn(i)) + 1;
    end
end