import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(columns=['calory', 'breakfast', 'lunch', 
                           'dinner', 'exercise', 'body_shape'])

df.loc[0] = [1200, 1, 0, 0, 2, 'Skinny']
df.loc[1] = [2800, 1, 1, 1, 1, 'Normal']
df.loc[2] = [3500, 2, 2, 1, 0, 'Fat']
df.loc[3] = [1400, 0, 1, 0, 3, 'Skinny']
df.loc[4] = [5000, 2, 2, 2, 0, 'Fat']
df.loc[5] = [1300, 0, 0, 1, 2, 'Skinny']
df.loc[6] = [3000, 1, 0, 1, 1, 'Normal']
df.loc[7] = [4000, 2, 2, 2, 0, 'Fat']
df.loc[8] = [2600, 0, 2, 0, 0, 'Normal']
df.loc[9] = [3000, 1, 2, 1, 1, 'Fat']

df.head()

X = df[['calory', 'breakfast', 'lunch', 'dinner', 'exercise']]
X.head()

Y = df[['body_shape']]
Y.head()

x_std = StandardScaler().fit_transform(X)
x_std

pca = decomposition.PCA(n_components=1)
sklearn_pca_x = pca.fit_transform(x_std)
sklearn_pca_x

sklearn_result = pd.DataFrame(sklearn_pca_x, columns = ['PC1'])
sklearn_result

sklearn_result['y-axis'] = 0.0
sklearn_result

sklearn_result['label'] = Y
sklearn_result

sns.lmplot('PC1', 'y-axis', data = sklearn_result, fit_reg = False,
           scatter_kws = {"s" : 50}, hue = "label")
