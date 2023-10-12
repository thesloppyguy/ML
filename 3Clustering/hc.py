from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("0DATA/Mall_Customers.csv")
x = dataset.iloc[:, [3, 4]].values

dendrogram = sch.dendrogram(sch.linkage(x, method="ward"))
plt.title("dendrogram")
plt.xlabel("customers points")
plt.ylabel("euclidian distance")
plt.show()

ac = AgglomerativeClustering(
    n_clusters=5, affinity="euclidean", linkage='ward')
cluster_list = ac.fit_predict(x)


plt.scatter(x[cluster_list == 0, 0], x[cluster_list == 0, 1],
            color='red', label="cluster-1")
plt.scatter(x[cluster_list == 1, 0], x[cluster_list == 1, 1],
            color='blue', label="cluster-2")
plt.scatter(x[cluster_list == 2, 0], x[cluster_list == 2, 1],
            color='purple', label="cluster-3")
plt.scatter(x[cluster_list == 3, 0], x[cluster_list == 3, 1],
            color='green', label="cluster-4")
plt.scatter(x[cluster_list == 4, 0], x[cluster_list == 4, 1],
            color='violet', label="cluster-5")
plt.title("CLUSTERS")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
