import numpy as np
import matplotlib.pyplot as plt

class KMeans():

    def __init__(self,k=3,maxiters=100,plot_steps=False):
        self.k = k
        self.maxiters = maxiters
        self.plot_steps = plot_steps

        self.centroids = np.empty((self.k,0))
        self.clusters = np.zeros(self.k)

    def fit(self,x):
        self.x = x
        self.n_samples,self.n_features = x.shape

        """Initialization Step"""
        random_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = self.x[random_idxs]
        print(random_idxs)

        for _ in range(self.maxiters):
            a = a+1

    def predict(self,x):
        self.x=x

    def get_cost(self,x):
        self.x=x