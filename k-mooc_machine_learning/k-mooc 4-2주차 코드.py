from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mnist = fetch_openml('mnist_784')
mnist_data = mnist.data[:10000]
mnist_target = mnist.target[:10000]

print(mnist_data)
print(mnist_data.shape)
print(mnist_target)
print(mnist_target.shape)

X_train, X_test, y_train, y_test = train_test_split(mnist_data, mnist_target, test_size = 0.2)

#파라미터 없이 학습하기
dt_clf = tree.DecisionTreeClassifier()
rf_clf = RandomForestClassifier()

dt_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)

dt_pred = dt_clf.predict(X_test)
rf_pred = rf_clf.predict(X_test)

accuracy_dt = accuracy_score(y_test, dt_pred)
accuracy_rf = accuracy_score(y_test, rf_pred)

print("의사결정 트리 예측 정확도: {0: .4f}".format(accuracy_dt))
print("랜덤 포레스트 예측 정확도: {0: .4f}".format(accuracy_rf))
print(rf_clf)

#특징 중요도 확인
ft_importances_values = rf_clf.feature_importances_
ft_importances = pd.Series(ft_importances_values)
top10 = ft_importances.sort_values(ascending=False)[:10]
plt.figure(figsize=(12,10))
plt.title('Feature Importances')
sns.barplot(x=top10.index, y=top10)
plt.show()

#파라미터를 사용하여 학습
rf_param_grid = {'n_estimators' :[100,110,120],
                 'min_samples_leaf' :[1, 2, 3],
                 'min_samples_split' :[2, 3, 4]
                 }
rf_clf = RandomForestClassifier(random_state = 0)
grid = GridSearchCV(rf_clf, param_grid = rf_param_grid,
                    scoring = 'accuracy', n_jobs = 1)
grid.fit(X_train,y_train)

print("최고 평균 정확도 :{0:.4f}".format(grid.best_score_))
print(grid.best_params_)
