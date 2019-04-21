addpath('..')
run('vl_setup')

numFeatures = 5000 ; % in fact, is the sample amount
dimension = 2 ; % feature dimensions
data = rand(dimension,numFeatures) ;

% accoring to AwA2 dataset, it should contains 50 clusters
numClusters = 50 ;
centers = vl_kmeans(data, numClusters);

kdtree = vl_kdtreebuild(centers) ;
nn = vl_kdtreequery(kdtree, centers, data) ;

assignments = zeros(numClusters,numFeatures);
% sub2ind() transforms sub-index to linear-index
% nn is the nearest cluster neighbor, points should be assigned to this cluster
assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;

% encoding (the length of encoding is 2*clusters)
encoding = vl_vlad(data,centers,assignments);