import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans

class ClusteringObject(ABC):
  
    def __init__(self, data : np.ndarray, n_clusters):
        self.data = data
        self.n_clusters = n_clusters

    @abstractmethod
    def get_labels(self):
        pass


class KMeansObject(ClusteringObject):
    
    def __init__(self, data: np.ndarray, n_clusters : int):
        super().__init__(data, n_clusters)
        self.KMeans = KMeans(self.n_clusters)
        self.KMeans.fit(self.data)
    
    def get_labels(self):
        return self.KMeans.labels_
    


