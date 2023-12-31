from sklearn import datasets
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
mnist = datasets.load_digits()
features, labels = mnist.data, mnist.target
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size= 0.2)

dtree = tree.DecisionTreeClassifier(criterion="gini", max_depth=8, max_features = 32)
dtree = dtree.fit(X_train, y_train)
dtree_predicted = dtree.predict(X_test)

knn=KNeighborsClassifier(n_neighbors = 299).fit(X_train, y_train)
knn_predicted = knn.predict(X_test)

svm = SVC(C=0.1, gamma = 0.003, probability=True).fit(X_train, y_train)
svm_predicted = svm.predict(X_test)

print(['accuracy'])
print("d-tree: ", accuracy_score(y_test, dtree_predicted))
print("knn: ", accuracy_score(y_test, knn_predicted))
print("svm: ", accuracy_score(y_test, svm_predicted))

svm_proba = svm.predict_proba(X_test)
print(svm_proba[0:2])

#하드보팅
voting_model = VotingClassifier(estimators = [
                                ('Decison_Tree', dtree),('k-NN',knn),('SVM',svm)],
                                weights = [1,1,1], voting='hard')
voting_model.fit(X_train, y_train)
hard_voting_predicted = voting_model.predict(X_test)
accuracy_score(y_test, hard_voting_predicted)

#소프트 보팅
voting_model = VotingClassifier(estimators =[('Decision_Tree', dtree), ('k-nn', knn), ('SVM', svm)],
                                weights=[1,1,1], voting='soft')
voting_model.fit(X_train, y_train)
soft_voting_predicted = voting_model.predict(X_test)
accuracy_score(y_test, soft_voting_predicted)

x = np.arange(5)
plt.bar(x, height = [accuracy_score(y_test, dtree_predicted),
                     accuracy_score(y_test, knn_predicted),
                     accuracy_score(y_test, svm_predicted),
                     accuracy_score(y_test, hard_voting_predicted),
                     accuracy_score(y_test, soft_voting_predicted)])
plt.xticks(x, ['decision tree', 'knn', 'svm', 'hard voting', 'soft voting'])
