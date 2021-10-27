from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
import matplotlib.pyplot as plt

df = pd.read_csv("./data/Ex06/train.csv", encoding='latin-1')

df['var3'].replace(-999999, 2, inplace=True)
df.drop('ID', axis=1, inplace=True)

X_features = df.iloc[:, :-1]
y_labels = df.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2)

# LightGBM 산탄데르 고객 만족 예측 : 학습
print("LightGBM 산탄데르 고객 만족 예측 : 학습 \n")

lgbm = LGBMClassifier(n_estimators=500)
evals = [(X_test, y_test)]
lgbm.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="auc", eval_set=evals, verbose=True)
pred_prob = lgbm.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, pred_prob, average='macro')
print("\n")

print("roc_score : ", roc_score)

print("\n")

# LightGBM 산탄데르 고객 만족 예측 : 파라미터 튜닝
print("LightGBM 산탄데르 고객 만족 예측 : 파라미터 튜닝 \n")

lgbm = LGBMClassifier(n_estimators=200)
params = {
    'num_leaves': [32, 64],
    'max_depth': [128, 160],
    'min_child_samples': [60, 100],
    'subsample': [0.8, 1]
}

grid_cv = GridSearchCV(lgbm, param_grid=params, cv=3)
grid_cv.fit(X_train, y_train, early_stopping_rounds=30, eval_metric="auc", eval_set=[(X_train, y_train), (X_test, y_test)])
print("grid_cv.best_estimator_ : ", grid_cv.best_estimator_)
print("\n")
pred_prob = grid_cv.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, pred_prob, average='macro')
print("roc_score : ", roc_score)

print("\n")

# LightGBM 산탄데르 고객 만족 예측 : 테스트
print("LightGBM 산탄데르 고객 만족 예측 : 테스트 \n")

lgbm = LGBMClassifier(n_estimators=1000, max_depth=128, min_child_weight=100, num_leaves=32, subsample=0.8)
evals = [(X_test, y_test)]
lgbm.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="auc", eval_set=evals, verbose=True)
pred_prob = lgbm.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, pred_prob, average='macro')
print("\n")
print("roc_score : ", roc_score)