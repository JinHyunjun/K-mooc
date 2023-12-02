from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd

iris = datasets.load_iris()
labels = pd.DataFrame(iris.target)
labels.columns = ['labels']
data = pd.DataFrame(iris.data)
data = pd.concat([data, labels], axis= 1)
data.head()

merge = linkage(data, method ='complete')

plt.figure(figsize=(30,10))
plt.title("IRIS Dendograms")

dendrogram(merge, leaf_rotation =90, leaf_font_size = 10)
plt.show()

cut = fcluster(merge, t=3, criterion = 'distance')

labels = data['labels']

df = pd.DataFrame({'predict': cut, 'labels': labels})
df

ct = pd.crosstab(df['predict'], df['labels'])
ct
