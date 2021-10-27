import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

print("사이킷런을 이용한 보스턴 주택 가격 예측")

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

print("\n")
print("df.shape : ", df.shape)
print("df.head : ", df.head())

print("\n")
print("df.info : ", df.info())

fig, axs = plt.subplots(figsize=(16, 8), ncols=4, nrows=2)
features = ['RM', 'ZN', 'INDUS', 'NOX', 'AGE', 'PTRATIO', 'LSTAT', 'RAD']
for i, feature in enumerate(features):
    row = int(i / 4)
    col = i % 4
    sns.regplot(x=feature, y='PRICE', data=df, ax=axs[row][col])

y = df['PRICE']
X = df.drop(['PRICE'], axis=1, inplace=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)
print("mse : %f, rmse : %f, r2 : %f " %(mse, rmse, r2))

print("\n")

print("lr.intercept_ : ", lr.intercept_)
print("lr.coef_ : ", lr.coef_)

coeff = pd.Series(data=lr.coef_, index=X.columns)

print("\n")

print(coeff.sort_values(ascending=False))

print("\n")

y = df['PRICE']
X = df.drop(['PRICE'], axis=1, inplace=False)
lr = LinearRegression()

neg_mse_scores = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=5)
rmse_scores = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

print("neg_mse_scores : ", neg_mse_scores)
print("rmse_scores : ", rmse_scores)
print("avg_rmse : ", avg_rmse)

print("\n")

X = np.arange(4).reshape(2, 2)
print("X : \n", X)
print("\n")

poly = PolynomialFeatures(degree=2)
poly.fit(X)
poly_ftr = poly.transform(X)
print("poly_ftr : \n", poly_ftr)

def polynomial_func(X):
    y = 1 + 2 * X[:, 0] + 3 * X[:, 0]**2 + 4 * X[:, 1]**3
    return y

print("\n")

X = np.arange(4).reshape(2, 2)
print("X : \n", X)

print("\n")
y = polynomial_func(X)
print("y : ", y)

print("\n")

poly = PolynomialFeatures(degree=3)
poly_ftr = poly.fit_transform(X)
print("poly_ftr : \n", poly_ftr)
print("\n")

lr = LinearRegression()
lr.fit(poly_ftr, y)
print("lr.coef_ : ", lr.coef_)
print("\n")
print("lr.coef_.shape : ", lr.coef_.shape)

plt.show()