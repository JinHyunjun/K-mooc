import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score

dataset = load_iris()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target
df.target = df.target.map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
df.head()

df.target.value_counts()

setosa_df = df[df.target=="setosa"]
versicolor_df = df[df.target == "versicolor"]
virginica_df = df[df.target== "virginica"]

ax = setosa_df['sepal length (cm)'].plot(kind='hist')
setosa_df['sepal length (cm)'].plot(kind = 'kde',
                                    ax = ax,
                                    secondary_y = True,
                                    title = "setosa sepal length",
                                    figsize = (8,4))

ax = versicolor_df['sepal length (cm)'].plot(kind='hist')
versicolor_df['sepal length (cm)'].plot(kind = 'kde',
                                    ax = ax,
                                    secondary_y = True,
                                    title = "versicolor sepal length",
                                    figsize = (8,4))

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = 0.2)
model = GaussianNB()
model.fit(X_train, y_train)

expected = y_test
predicted = model.predict(X_test)

print(metrics.classification_report(y_test,predicted))
print(accuracy_score(y_test,predicted))

print(metrics.confusion_matrix(expected,predicted))