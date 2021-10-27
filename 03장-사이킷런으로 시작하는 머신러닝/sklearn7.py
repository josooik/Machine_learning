from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)
iris_data = iris.data
iris_label = iris.target

scores = cross_val_score(dt_clf, iris_data, iris_label, scoring='accuracy', cv=3)
print(np.round(scores, 4))
print(np.round(np.mean(scores), 4))