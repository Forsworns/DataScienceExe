from sklearn.neighbors import DistanceMetric as Dist
import numpy as np

class Distance:
    def __init__(self,X,M):
        self.X = X
        self.M = M

    def euclidean(self):
        return 'euclidean'

    def manhattan(self):
        return 'manhattan'

    def chebyshev(self):
        return 'chebyshev'

    def minkowski(self):
        # p = 1 euclidean, p = 2 manhattan, p = inf chebyshev
        return 'minkowski'

    def mahalanobis(self):
        return 'mahalanobis'

    def canberra(self):
        return 'canberra'

    def braycurtis(self):
        return 'braycurtis'

    def cosine(self):
        return lambda x,y:np.dot(x,y)/np.linalg.norm(x,ord=2)/np.linalg(y,ord=2)
