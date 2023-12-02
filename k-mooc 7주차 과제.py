import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv("C:/Users/SAMSUNG/Desktop/heart_disease_uci.csv")

train, test = train_test_split(df, test_size = 0.2)

#support vector machine

def svc_param_selection(X, y, nfolds):
    svm_parameters = [{'kernel' :['rbf'],
                      'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
                      'C': [0.01, 0.1, 1, 10, 100, 1000]}]
    clf = GridSearchCV(SVC(), svm_parameters, cv=nfolds)
    clf.fit(X, y)
    print(clf.best_params_)
    return clf

X_train = train[['sex', 'trestbps', 'chol']]
y_train = train[['target']]
clf = svc_param_selection(X_train, y_train.values.ravel(), 10)

X_test = test[['sex', 'trestbps', 'chol']]
y_test = test [['target']]

y_true, y_pred = y_test, clf.predict(X_test)

print(classification_report(y_true, y_pred))
print()
print("accuracy: " +str(accuracy_score(y_true, y_pred)))

# k-nn 알고리즘
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train.values.ravel())

y1_pred = knn_clf.predict(X_test)
print("k-nn accuracy: {0: .4f}".format(accuracy_score(y_test, y1_pred)))