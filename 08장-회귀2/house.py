import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("데이터 전처리 \n")

df_org = pd.read_csv('./data/L08_csv/house_prices/train.csv')
df = df_org.copy()
print("df.head \n", df.head(3))
print("df.shape \n", df.shape)
print("\n")
print("df.dtypes.value_counts \n", df.dtypes.value_counts())
print("\n")
isnull_series = df.isnull().sum()
print("isnull_series \n", isnull_series[isnull_series > 0].sort_values(ascending=False))
print("\n")

sns.distplot(df['SalePrice'])
plt.show()

log_saleprice = np.log1p(df['SalePrice'])
sns.distplot(log_saleprice)
plt.show()

original_saleprice = df['SalePrice']
df['SalePrice'] = np.log1p(df['SalePrice'])
print(df.head(3))
print("\n")

df.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=True)
df.fillna(df.mean(), inplace=True)
print(df.head(3))
print("\n")

null_column_count = df.isnull().sum()[df.isnull().sum() > 0]
print("df.dtypes \n", df.dtypes[null_column_count.index])
print("\n")

print("df.shape \n", df.shape)
print("\n")

df_ohe = pd.get_dummies(df)
print("df_ohe.shape \n", df_ohe.shape)
print("\n")

null_column_count = df_ohe.isnull().sum()[df_ohe.isnull().sum() > 0]
print("df_ohe.dtypes \n", df_ohe.dtypes[null_column_count.index])
print("\n")