from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/wikibook/machine-learning/2.0/data/csv/basketball_stat.csv")
train, test = train_test_split(df, test_size = 0.2)

def svc_param_selection(X, y, nfolds):
    svm_parameters = [{'kernel':['rbf'],
                      'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
                      'C': [0.01, 0.1, 1, 10, 100, 1000]}]
    clf = GridSearchCV(SVC(), svm_parameters, cv = nfolds)
    clf.fit(X,y)
    print(clf.best_params_)
    return clf

X_train = train[['3P', 'BLK']]
y_train = train[['Pos']]
clf = svc_param_selection(X_train, y_train.values.ravel(), 10)

X_test = test[['3P', 'BLK']]
y_test = test[['Pos']]

y_true, y_pred = y_test, clf.predict(X_test)

print(classification_report(y_true, y_pred))
print()
print("accuracy: "+str(accuracy_score(y_true, y_pred)))

comparison = pd.DataFrame({'prdiction': y_pred, 'ground_truth': y_true.values.ravel()})
print(comparison)