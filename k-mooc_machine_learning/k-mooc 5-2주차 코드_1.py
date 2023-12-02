import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns

df = pd.read_csv('C:/Users/SAMSUNG/Desktop/kmeans.csv')
print(df.head())
print(df.shape)


sns.lmplot('height', 'weight', data = df,
           fit_reg = False,
           scatter_kws = {'s':30})

data_points = df.values
kmeans = KMeans(n_clusters=3).fit(data_points)

kmeans.labels_[:10]

df['cluster'] = kmeans.labels_
df.head()

sns.lmplot('height', 'weight', data = df,
           fit_reg = False,
           scatter_kws = {'s':30},
           hue = 'cluster')
