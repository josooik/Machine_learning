import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import skew
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

print("데이터 전처리 \n")

df_org = pd.read_csv('./data/L08_csv/house_prices/train.csv')
df = df_org.copy()

log_saleprice = np.log1p(df['SalePrice'])

original_saleprice = df['SalePrice']
df['SalePrice'] = np.log1p(df['SalePrice'])

df.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=True)
df.fillna(df.mean(), inplace=True)

null_column_count = df.isnull().sum()[df.isnull().sum() > 0]

df_ohe = pd.get_dummies(df)

null_column_count = df_ohe.isnull().sum()[df_ohe.isnull().sum() > 0]

print("학습 및 평가 \n")

def get_rmse(model):
    pred = model.predict(x_test)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    print("rmse : ", rmse)
    return rmse

def get_rmses(models):
    rmses = []
    for model in models:
        rmse = get_rmse(model)
        rmses.append(rmse)
    return rmses

y_target = df_ohe['SalePrice']
x_features = df_ohe.drop('SalePrice', axis=1, inplace=False)
x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size=0.2)

lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()

lr.fit(x_train, y_train)
ridge.fit(x_train, y_train)
lasso.fit(x_train, y_train)

models = [lr, ridge, lasso]
get_rmses(models)

def get_top_bottom_coef(model, n=10):
    coef = pd.Series(model.coef_, index=x_features.columns)
    coef_high = coef.sort_values(ascending=False).head(n)
    coef_low = coef.sort_values(ascending=False).tail(n)
    return coef_high, coef_low

def visualize_coeff(models):
    fig, axs = plt.subplots(figsize=(24, 10), nrows=1, ncols=3)
    for i, model in enumerate(models):
        coef_high, coef_low = get_top_bottom_coef(model)
        coef_concat = pd.concat([coef_high, coef_low])
        sns.barplot(x=coef_concat.values, y=coef_concat.index, ax=axs[i])

models = [lr, ridge, lasso]
visualize_coeff(models)

print("\n")

print("cross_val_score \n")

def get_avg_rmse_cv(models):
    for model in models:
        rmse_list = np.sqrt(-cross_val_score(model, x_features, y_target, scoring='neg_mean_squared_error', cv=5))
        rmse_avg = np.mean(rmse_list)
        print("rmse_list : ", rmse_list)
        print("rmse_avg : ", rmse_avg)
        print("\n")

models = [lr, ridge, lasso]
get_avg_rmse_cv(models)

print("GridSearchCV \n")

def print_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid=params, scoring='neg_mean_squared_error', cv=5)
    grid_model.fit(x_features, y_target)
    rmse = np.sqrt(-1 * grid_model.best_score_)
    print("rmse : ", rmse)
    print("grid_model.best_params_ : ", grid_model.best_params_)
    print("\n")

ridge_params = {
    'alpha' : [0.05, 0.1, 1, 5, 8, 10, 12, 15, 20]
}
lasso_params = {
    'alpha' : [0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1, 5, 10]
}

print_best_params(ridge, ridge_params)
print_best_params(lasso, lasso_params)

lr = LinearRegression()
ridge = Ridge(alpha=12)
lasso = Lasso(alpha=0.001)

lr.fit(x_train, y_train)
ridge.fit(x_train, y_train)
lasso.fit(x_train, y_train)

models = [lr, ridge, lasso]
get_rmses(models)
visualize_coeff(models)

print("\n")

print("skew \n")

features_index = df.dtypes[df.dtypes != 'object'].index
skew_features = df[features_index].apply(lambda  x: skew(x))
skew_features_top = skew_features[skew_features > 1]
print(skew_features_top.sort_values(ascending=False))

print("\n")

df[skew_features_top.index] = np.log1p(df[skew_features_top.index])
df_ohe = pd.get_dummies(df)
y_target = df_ohe['SalePrice']
x_features = df_ohe.drop('SalePrice', axis=1, inplace=False)
x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size=0.2)

ridge_params = {
    'alpha' : [0.005, 0.1, 1, 5, 8, 10, 12, 15, 20]
}
lasso_params = {
    'alpha' : [0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1, 5, 10]
}

print_best_params(ridge, ridge_params)
print_best_params(lasso, lasso_params)
plt.show()

plt.scatter(x=df_org['GrLivArea'], y=df_org['SalePrice'])
plt.ylabel('SalePrifce')
plt.xlabel('GrLivArea')
plt.show()

cond1 = df_ohe['GrLivArea'] > np.log1p(4000)
cond2 = df_ohe['SalePrice'] < np.log1p(500000)
outtlier_index = df_ohe[cond1 & cond2].index

print("outtlier_index : ", outtlier_index)
print("df_ohe.shape : ", df_ohe.shape)
df_ohe.drop(outtlier_index, axis=0, inplace=True)
print("df_ohe.shape : ", df_ohe.shape)

plt.scatter(x=df_ohe['GrLivArea'], y=df_ohe['SalePrice'])
plt.ylabel('SalePrifce')
plt.xlabel('GrLivArea')
plt.show()

print("\n")

y_target = df_ohe['SalePrice']
x_features = df_ohe.drop('SalePrice', axis=1, inplace=False)
x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size=0.2)

ridge_params = {
    'alpha' : [0.05, 0.1, 1, 5, 8, 10, 12, 15, 20]
}
lasso_params = {
    'alpha' : [0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1, 5, 10]
}

print_best_params(ridge, ridge_params)
print_best_params(lasso, lasso_params)

lr = LinearRegression()
ridge = Ridge(alpha=8)
lasso = Lasso(alpha=0.001)

lr.fit(x_train, y_train)
ridge.fit(x_train, y_train)
lasso.fit(x_train, y_train)

models = [lr, ridge, lasso]
get_rmses(models)
visualize_coeff(models)
plt.show()

print("\n")

print("회귀 트리 \n")

print("XGBRegressor \n")

xgb_params = {
    'n_estimators' : [1000]
}

xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, colsample_bytree=0.5, subsample=0.8)
print_best_params(xgb, xgb_params)

print("LGBMRegressor \n")

lgbm_params = {
    'n_estimators' : [1000]
}

lgbm = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=4, subsample=0.6, colsample_bytree=0.4, reg_lambda=10, n_jobs=-1)
print_best_params(lgbm, lgbm_params)

print("예측 결과 혼합 \n")

def get_rmse_pred(preds):
    for key in preds.keys():
        pred_value = preds[key]
        mse = mean_squared_error(y_test, pred_value)
        rmse = np.sqrt(mse)
        print(key, rmse)

ridge = Ridge(alpha=8)
lasso = Lasso(alpha=0.001)

ridge.fit(x_train, y_train)
lasso.fit(x_train, y_train)

ridge_pred = ridge.predict(x_test)
lasso_pred = lasso.predict(x_test)

pred = 0.4 * ridge_pred + 0.6 * lasso_pred

preds = {
    '최종 혼합 : ' : pred,
    'Ridge : ': ridge_pred,
    'Lasso : ': lasso_pred
}

get_rmse_pred(preds)

print("\n")

xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, colsample_bytree=0.5, subsample=0.8)
lgbm = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=4, subsample=0.6, colsample_bytree=0.4, reg_lambda=10, n_jobs=-1)

xgb.fit(x_train, y_train)
lgbm.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)
lgbm_pred = lgbm.predict(x_test)

pred = 0.5 * xgb_pred + 0.5 * lgbm_pred
preds = {
    '최종 혼합 : ' : pred,
    'XGB : ' : xgb_pred,
    'LGBM : ' : lgbm_pred
}

get_rmse_pred(preds)

print("\n")

print("스태킹 앙상블 \n")

def get_stacking_base_datasets(model, X_train, y_train, X_test, n_folds):
    kf = KFold(n_splits=n_folds, shuffle=False)
    train_fold_pred = np.zeros((X_train.shape[0], 1))
    test_pred = np.zeros((X_test.shape[0], n_folds))

    for folder_index, (train_index, valid_indx) in enumerate(kf.split(X_train)):
        X_tr = X_train[train_index]
        y_tr = y_train[train_index]
        X_te = X_train[valid_indx]
        model.fit(X_tr, y_tr)
        train_fold_pred[valid_indx, :] = model.predict(X_te).reshape(-1, 1)
        test_pred[:, folder_index] = model.predict(X_test)

    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)
    return train_fold_pred, test_pred_mean

x_train_values = x_train.values
x_test_values = x_test.values
y_train_values = y_train.values

ridge_train, ridge_test = get_stacking_base_datasets(ridge, x_train_values, y_train_values, x_test_values, 5)
lasso_train, lasso_test = get_stacking_base_datasets(lasso, x_train_values, y_train_values, x_test_values, 5)
xgb_train, xgb_test = get_stacking_base_datasets(xgb, x_train_values, y_train_values, x_test_values, 5)
lgbm_train, lgbm_test = get_stacking_base_datasets(lgbm, x_train_values, y_train_values, x_test_values, 5)

final_x_train = np.concatenate((ridge_train, lasso_train, xgb_train, lgbm_train), axis=1)
final_x_test = np.concatenate((ridge_test, lasso_test, xgb_test, lgbm_test), axis=1)
meta_model = Lasso(alpha=0.0005)
meta_model.fit(final_x_train, y_train)
final_pred = meta_model.predict(final_x_test)
mse = mean_squared_error(y_test, final_pred)
rmse = np.sqrt(mse)
print("rmse : ", rmse)