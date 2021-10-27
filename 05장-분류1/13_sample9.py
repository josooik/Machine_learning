import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# XGBoost (eXtra Gradient Boost)
print("XGBoost (eXtra Gradient Boost)")
print("\n")

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

pred_prob = xgb_model.predict(dtest)
print(pred_prob[:10])

print("\n")

preds = [1 if x > 0.5 else 0 for x in pred_prob]
print(preds[:10])

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

get_clf_eval(y_test, preds, pred_prob)

fix, ax = plt.subplots(figsize=(10, 12))
plot_importance(xgb_model, ax=ax)
plt.show()