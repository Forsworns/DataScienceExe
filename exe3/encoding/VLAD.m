numClusters = 30 ;
centers = vl_kmeans(dataLearn, numClusters);

kdtree = vl_kdtreebuild(centers) ;
nn = vl_kdtreequery(kdtree, centers, dataEncode) ;

assignments = zeros(numClusters,numDataToBeEncoded);
assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;

enc = vl_vlad(dataToBeEncoded,centers,assignments);