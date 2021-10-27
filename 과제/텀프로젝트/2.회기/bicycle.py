import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

df = pd.read_csv('./data/L08_csv/bike_sharing/train.csv')
print("df.shape : ", df.shape)
print("\n", df.head())
print("\n", df.info())

print("데이터 전처리 \n")

df['datetime'] = df.datetime.apply(pd.to_datetime)
df['year'] = df.datetime.apply(lambda x: x.year)
df['month'] = df.datetime.apply(lambda x: x.month)
df['day'] = df.datetime.apply(lambda x: x.day)
df['hour'] = df.datetime.apply(lambda x: x.hour)
print(df.head(3))

drop_columns = ['datetime', 'casual', 'registered']
df.drop(drop_columns, axis=1, inplace=True)

print("\n")

print("로그 변환(expm1(x) = exp(x) - 1) \n")

def rmsle(y, pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred) ** 2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle

def rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))

def evaluate_regr(y, pred):
    rmsle_val = rmsle(y, pred)
    rmse_val = rmse(y, pred)
    mae_val = mean_absolute_error(y, pred)
    print("rmsle_val : %f, rmse_val : %f, mae_val : %f" %(rmsle_val, rmse_val, mae_val))

y_target = df['count']
x_features = df.drop(['count'], axis=1, inplace=False)

y_log = np.log1p(y_target)

X_train, x_test, y_train, y_test = train_test_split(x_features, y_log, test_size=0.3)

lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(x_test)
y_test_exp = np.expm1(y_test)
pred_exp = np.expm1(pred)
evaluate_regr(y_test_exp, pred_exp)

print("\n")

print("회귀 계수 값 시각화 \n")

coef = pd.Series(lr.coef_, index=x_features.columns)
coef_sort = coef.sort_values(ascending=False)
sns.barplot(x=coef_sort.values, y=coef_sort.index)
plt.show()

print("카테고리형 피처  원핫 인코딩 \n")

x_features_ohe = pd.get_dummies(x_features, columns=['year', 'month', 'day', 'hour', 'holiday', 'workingday', 'season', 'weather'])
x_train, x_test, y_train, y_test = train_test_split(x_features_ohe, y_log, test_size=0.3)

def get_model_predict(model, x_train, x_test, y_train, y_test, is_expm1=False):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    if is_expm1:
        y_test = np.expm1(y_test)
        pred = np.expm1(pred)
    evaluate_regr(y_test, pred)

lr = LinearRegression()
ridge = Ridge(alpha=10)
lasso = Lasso(alpha=0.01)

for model in [lr, ridge, lasso]:
    get_model_predict(model, x_train, x_test, y_train, y_test, is_expm1=True)

print("\n")

print("회귀 계수 값 시각화 \n")

coef = pd.Series(lr.coef_, index=x_features_ohe.columns)
coef_sort = coef.sort_values(ascending=False)[:20]
sns.barplot(x=coef_sort.values, y=coef_sort.index)
plt.show()

print("모델 회기 트리 \n")

rf = RandomForestRegressor(n_estimators=500)
gbm = GradientBoostingRegressor(n_estimators=500)
xgb = XGBRegressor(n_estimators=500)
lgbm = LGBMRegressor(n_estimators=500)

for model in [rf, gbm, xgb, lgbm]:
    get_model_predict(model, x_train.values, x_test.values, y_train.values, y_test.values, is_expm1=True)