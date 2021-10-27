from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from lightgbm import plot_importance
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2)
lgbm = LGBMClassifier(n_estimators=400)
evals = [(X_test, y_test)]
lgbm.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", eval_set=evals, verbose=True)
pred = lgbm.predict(X_test)
pred_prob = lgbm.predict_proba(X_test)[:,1]

print("\n")

def get_clf_eval(y_test, pred=None, pred_prob=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_prob)

    print("confusion : ", confusion)
    print("accuracy : ", accuracy)
    print("precision : ", precision)
    print("recall : ", recall)
    print("f1 : ", f1)
    print("roc_auc : ", roc_auc)

get_clf_eval(y_test, pred, pred_prob)

fix, ax = plt.subplots(figsize=(10, 12))
plot_importance(lgbm, ax=ax)
plt.show()