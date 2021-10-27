from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
iris_data = iris.data
iris_label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

kfold = KFold(n_splits = 5)
cv_accuracy = []
n_iter = 0

for train_index, test_index in kfold.split(iris_data):
    X_train, X_test = iris_data[train_index], iris_data[test_index]
    y_train, y_test = iris_label[train_index], iris_label[test_index]
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    n_iter += 1
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print(n_iter, accuracy, train_size, test_size)
    print(n_iter, test_index)
    cv_accuracy.append(accuracy)

print(np.mean(cv_accuracy))