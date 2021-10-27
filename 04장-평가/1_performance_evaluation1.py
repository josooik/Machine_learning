import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class MyFakeClassifier(BaseEstimator):
    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros((X.shape[0], 1), dtype=int)

digits = load_digits()
y = (digits.target == 7).astype(int)

X_train, X_test, y_train, y_test = train_test_split(digits.data, y)
print(y_test.shape)
print(pd.Series(y_test).value_counts())

clf = MyFakeClassifier()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print("Accuracy : ", accuracy_score(y_test, predictions))

print("\n")

print("confusion_matrix : \n", confusion_matrix(y_test, predictions))
