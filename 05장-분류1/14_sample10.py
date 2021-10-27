import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# 사이킷런 래퍼 XGBoost (eXtra Gradient Boost)
print("사이킷런 래퍼 XGBoost (eXtra Gradient Boost)")
print("\n")

def get_clf_eval(y_test, pred=None, pred_prob=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_prob)

    print("confusion : \n", confusion)
    print("accuracy : ", accuracy)
    print("precision : ", precision)
    print("recall : ", recall)
    print("f1 : ", f1)
    print("roc_auc : ", roc_auc)

cancer = load_breast_cancer()
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target
print(df.head(3))

print(cancer.target_names)
print(df['target'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2)
print(X_train.shape, X_test.shape)

dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)
params = {
    'max_depth' : 3,
    'eta' : 0.1,
    'objective' : 'binary:logistic',
    'eval_metric' : 'logloss',
    'early_stoppings' : 100
}

num_rounds = 400

wlist = [(dtrain, 'train'), (dtest, 'eval')]
xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds, early_stopping_rounds=100, evals=wlist)

print("\n")

xgb = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
xgb.fit(X_train, y_train)
pred = xgb.predict(X_test)
pred_prob = xgb.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, pred, pred_prob)

print("\n")

xgb = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
evals = [(X_test, y_test)]
xgb.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='logloss', eval_set=evals)
pred100 = xgb.predict(X_test)
pred_prob100 = xgb.predict_proba(X_test)[:, 1]

get_clf_eval(y_test, pred100, pred_prob100)

print("\n")

xgb.fit(X_train, y_train, early_stopping_rounds=10, eval_metric='logloss', eval_set=evals)
pred10 = xgb.predict(X_test)
pred_prob10 = xgb.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, pred10, pred_prob10)

fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(xgb_model, ax=ax)
plt.show()