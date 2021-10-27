from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target
y_target = df['PRICE']
x_data = df.drop(['PRICE'], axis=1, inplace=False)

print("회귀 트리 \n")

rf = RandomForestRegressor(n_estimators=1000)
neg_mse_scores = cross_val_score(rf, x_data, y_target, scoring="neg_mean_squared_error", cv=5)
rmse_scores = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

print("neg_mse_scores : ", neg_mse_scores)
print("rmse_scores : ", rmse_scores)
print("avg_rmse : ", avg_rmse)

print("\n")

def get_model_cv_prediction(model, X_data, y_target):
    neg_mse_scores = cross_val_score(model, X_data, y_target, scoring="neg_mean_squared_error", cv=5)
    rmse_scores = np.sqrt(-1 * neg_mse_scores)
    avg_rmse = np.mean(rmse_scores)
    print("avg_rmse : ", avg_rmse)

dt = DecisionTreeRegressor(max_depth=4)
rf = RandomForestRegressor(n_estimators=1000)
gb = GradientBoostingRegressor(n_estimators=1000)
xgb = XGBRegressor(n_estimators=1000)
lgb = LGBMRegressor(n_estimators=1000)

models = [dt, rf, gb, xgb, lgb]
for model in models:
    get_model_cv_prediction(model, x_data, y_target)

rf = RandomForestRegressor(n_estimators=1000)
rf.fit(x_data, y_target)
feature = pd.Series(data=rf.feature_importances_, index=x_data.columns)
feature = feature.sort_values(ascending=False)
sns.barplot(x=feature, y=feature.index)
plt.show()

print("\n")

df_sample = df[['RM', 'PRICE']]
df_sample = df_sample.sample(n=100)
print("df_sample.shape : ", df_sample.shape)
plt.figure()
plt.scatter(df_sample.RM, df_sample.PRICE, c='darkorange')
plt.show()

print("\n")

lr = LinearRegression()
dt2 = DecisionTreeRegressor(max_depth=2)
dt7 = DecisionTreeRegressor(max_depth=7)

x_test = np.arange(4.5, 8.5, 0.04).reshape(-1, 1)
x_feature = df_sample['RM'].values.reshape(-1, 1)
y_target = df_sample['PRICE'].values.reshape(-1, 1)

lr.fit(x_feature, y_target)
dt2.fit(x_feature, y_target)
dt7.fit(x_feature, y_target)

pred_lr = lr.predict(x_test)
pred_dt2 = dt2.predict(x_test)
pred_dt7 = dt7.predict(x_test)

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(14, 4), ncols=3)
ax1.set_title('Linear Regression')
ax1.scatter(df_sample.RM, df_sample.PRICE, c='darkorange')
ax1.plot(x_test, pred_lr, label='linear', linewidth=2)

ax2.set_title('Decision Tree Regression : max_depth=2')
ax2.scatter(df_sample.RM, df_sample.PRICE, c='darkorange')
ax2.plot(x_test, pred_dt2, label='max_depth : 2', linewidth=2)

ax3.set_title('Decision Tree Regression : max_depth=7')
ax3.scatter(df_sample.RM, df_sample.PRICE, c='darkorange')
ax3.plot(x_test, pred_dt7, label='max_depth : 7', linewidth=2)
plt.show()