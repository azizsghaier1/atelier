import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import math



# Définition de classe
class K_Means:

    def __init__(self, k=2, eps = 0.001, max_iter = 500):
        self.k = k
        self.max_iterations = max_iter
        self.eps = eps

    def distance_euc(self, point1, point2):
        #return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)   #sqrt((x1-x2)^2 + (y1-y2)^2)
        return np.linalg.norm(point1-point2, axis=0)

    def predict(self,data):
        distances = [np.linalg.norm(data-self.barys[bary]) for bary in self.barys]
        classification = distances.index(min(distances))
        return classification

    def fit(self, data):
        self.barys = {}
        for i in range(self.k):
            self.barys[i] = data[i]


        for i in range(self.max_iterations):
            self.classes = {}
            for j in range(self.k):
                self.classes[j] = []

            for point in data:
                distances = []
                for index in self.barys:
                    distances.append(self.distance_euc(point,self.barys[index]))
                cluster_index = distances.index(min(distances))
                self.classes[cluster_index].append(point)

            prec = dict(self.barys)
            for cluster_index in self.classes:
                self.barys[cluster_index] = np.average(self.classes[cluster_index], axis = 0)



            isOptimal = True

            for bary in self.barys:
                original_bary = prec[bary]
                curr = self.barys[bary]
                if np.sum((curr - original_bary)/original_bary * 100.0) > self.eps:
                    isOptimal = False
            if isOptimal:
                break

K=3  #nombre de cluster
center_1 = np.array([1,1])
center_2 = np.array([5,5])
center_3 = np.array([8,1])

# Generate random data and center it to the three centers
cluster_1 = np.random.randn(100, 2) + center_1
cluster_2 = np.random.randn(100,2) + center_2
cluster_3 = np.random.randn(100,2) + center_3

data = np.concatenate((cluster_1, cluster_2, cluster_3), axis =0)


k_means = K_Means(K)
k_means.fit(data)


# Plotting starts here
colors = 10*["r", "g", "c", "b", "k"]

for bary in k_means.barys:
    plt.scatter(k_means.barys[bary][0], k_means.barys[bary][1], s = 130, marker = "x")

for indice_cluster in k_means.classes:
    color = colors[indice_cluster]
    for features in k_means.classes[indice_cluster]:
        plt.scatter(features[0], features[1], color = color,s = 30)
