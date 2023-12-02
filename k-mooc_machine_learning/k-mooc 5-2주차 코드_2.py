import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data[:,[0,1]]
data[:10]

kmeans_iris = KMeans(n_clusters = 3).fit(data)

labels = kmeans_iris.labels_
plt.title('Clustering Result', fontsize = 20)
plt.scatter(data[:,0], data[:,1], c=labels, s=60)

target = iris.target
df = pd.DataFrame({'labels': labels, 'target': target})
ct = pd.crosstab(df['labels'], df['target'])
ct

num_clusters = list(range(2,9))
inertias = []

for i in num_clusters:
    model = KMeans(n_clusters=i)
    model.fit(data)
    inertias.append(model.inertia_)
    
plt.plot(num_clusters, inertias, '-o')
plt.xlabel('Num of clusters')
plt.ylabel('inertia')
plt.show()
