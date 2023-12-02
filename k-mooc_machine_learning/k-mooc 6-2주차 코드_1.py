from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/SAMSUNG/Desktop/LinearRegression.csv")
df.head()
df.shape

X = df['height']
Y = df['weight']

lr = LinearRegression()
lr.fit(X.values.reshape(-1,1), Y)

print(lr.coef_)
print(lr.intercept_)

plt.plot(X, Y, 'o')
plt.plot(X, lr.predict(X.values.reshape(-1,1)))
plt.show()

df['height'] = df['height'].astype(float)
df['weight'] = df['weight'].astype(float)

sns.lmplot(x='height', y='weight', data = df, size = 7, line_kws={'color' : "red"})
