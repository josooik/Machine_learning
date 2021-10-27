import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
import matplotlib.pyplot as plt

df = pd.read_csv("./data/Ex06/train.csv", encoding='latin-1')
print("df_shaple : ", df.shape)
print("df_head : ", df.head(3))
print("df_info : ", df.info())

print("\n")

print(df['TARGET'].value_counts())
unsatisfied_cnt = df[df['TARGET'] == 1].TARGET.count()
total_cnt = df.TARGET.count()
print("평균 : ", unsatisfied_cnt / total_cnt)

print("\n")

print(df.describe())

df['var3'].replace(-999999, 2, inplace=True)
df.drop('ID', axis=1, inplace=True)

X_features = df.iloc[:, :-1]
y_labels = df.iloc[:, -1]
print("X_features_shape", X_features.shape)

print("\n")

print(df.describe())

print("\n")

X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2)

print("X_train_shape, X_test_shape : ", X_train.shape, X_test.shape)

train_cnt = y_train.count()
test_cnt = y_test.count()

print("y_train 비율 : \n", y_train.value_counts() / train_cnt)
print("\n")
print("y_test 비율 : \n", y_test.value_counts() / test_cnt)

print("\n")

# XGBoost 산탄데르 고객 만족 예측
print("XGBoost 산탄데르 고객 만족 예측 \n")

xgb = XGBClassifier(n_estimators=500)
xgb.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="auc", eval_set=[(X_train, y_train), (X_test, y_test)])
pred_prob = xgb.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, pred_prob, average='macro')
print("roc_score : ", roc_score)

print("\n")

# XGBoost 산탄데르 고객 만족 예측 : 파라미터 튜닝
print("XGBoost 산탄데르 고객 만족 예측 : 파라미터 튜닝 \n")

xgb = XGBClassifier(n_estimators=100)
params = {
    'max_depth' : [5, 7],
    'min_child_weight' : [1, 3],
    'colsample_bytree' : [0.5, 0.75]
}

grid_cv = GridSearchCV(xgb, param_grid=params, cv=3)
grid_cv.fit(X_train, y_train, early_stopping_rounds=30, eval_metric="auc", eval_set=[(X_train, y_train), (X_test, y_test)])
print("grid_cv.best_estimator_ : ", grid_cv.best_estimator_)
pred_prob = grid_cv.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, pred_prob, average='macro')
print("roc_score : ", roc_score)

print("\n")

# XGBoost 산탄데르 고객 만족 예측 : 테스트
print("XGBoost 산탄데르 고객 만족 예측 : 테스트 \n")

xgb = XGBClassifier(n_estimators=1000, learning_rate=0.002, reg_alpha=0.03, colsample_byree=1, max_depth=5, min_child_weight=3)
xgb.fit(X_train, y_train, early_stopping_rounds=200, eval_metric="auc", eval_set=[(X_train, y_train), (X_test, y_test)])
pred_prob = xgb.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, pred_prob, average='macro')
print("roc_score : ", roc_score)

print("\n")

# XGBoost 산탄데르 고객 만족 예측 : 피처 중요도
print("XGBoost 산탄데르 고객 만족 예측 : 피처 중요도 \n")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_importance(xgb, ax=ax, max_num_features=20, height=0.4)

plt.show()