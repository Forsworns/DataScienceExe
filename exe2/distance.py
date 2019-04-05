import sklearn.neighbors.DistanceMetric as Dist
import numpy as np

class Distance:
    def __init__(self,X,M):
        self.X = X
        self.M = M

    def euclidean(self):
        dist = Dist.get_metric('euclidean')
        return dist.pairwise(self.X)

    def manhattan(self):
        dist = Dist.get_metric('manhattan')
        return dist.pairwise(self.X)

    def chebyshev(self):
        dist = Dist.get_metric('chebyshev')
        return dist.pairwise(self.X)

    def minkowski_3(self):
        # p = 1 euclidean, p = 2 manhattan, p = inf chebyshev
        dist = Dist.get_metric('minkowski',3)
        return dist.pairwise(self.X)

    def minkowski_4(self):
        # p = 1 euclidean, p = 2 manhattan, p = inf chebyshev
        dist = Dist.get_metric('minkowski',4)
        return dist.pairwise(self.X)

    def mahalanobis(self):
        dist = Dist.get_metric('mahalanobis',self.M)
        return dist.pairwise(self.X)

    def canberra(self):
        dist = Dist.get_metric('canberra')
        return dist.pairwise(self.X)

    def braycurtis(self):
        dist = Dist.get_metric('braycurtis')
        return dist.pairwise(self.X)

    def cosine(self):
        dist = Dist.get_metric('pyfunc',lambda x,y:np.dot(x,y)/np.linalg.norm(x,ord=2)/np.linalg(y,ord=2))
        return dist.pairwise(self.X)
