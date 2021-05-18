from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("DATA/Mall_Customers.csv")
# can have multiple features but we are using two for visual purposes
x = dataset.iloc[:, [3, 4]].values

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init="k-means++", random_state=42)
    km.fit(x)
    wcss.append(km.inertia_)

# plt.plot(range(1, 11), wcss)
# plt.title("Elbow Method")
# plt.xlabel("clusters")
# plt.ylabel("wcss")
# plt.show()

km = KMeans(n_clusters=5, init="k-means++", random_state=42)
cluster_list = km.fit_predict(x)


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
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1], color="yellow")
plt.title("CLUSTERS")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
