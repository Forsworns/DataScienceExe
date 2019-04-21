addpath('..')
run('vl_setup')

numFeatures = 5000 ;
dimension = 2 ;
data = rand(dimension,numFeatures) ;

% train a GMM
numClusters = 50 ;
[means, covariances, priors] = vl_gmm(data, numClusters);

% ecoding by GMM (encoding length is 4*numclusters?)
encoding = vl_fisher(data, means, covariances, priors);