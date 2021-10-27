import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt

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

print("로그 변환 \n")

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
x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size=0.3)

lr = LinearRegression()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
evaluate_regr(y_test, pred)

print("\n")

def get_top_error_data(y_test, pred, n_tops=5):
    result_df = pd.DataFrame(y_test.values, columns=['real_count'])
    result_df['predicted_count'] = np.round(pred)
    result_df['diff'] = np.abs(result_df['real_count'] - result_df['predicted_count'])
    print(result_df.sort_values('diff', ascending=False)[:n_tops])

get_top_error_data(y_test, pred)

print("\n")

y_target.hist()
plt.show()

y_log = np.log1p(y_target)
y_log.hist()
plt.show()