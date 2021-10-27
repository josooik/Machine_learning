from sklearn.preprocessing import MinMaxScaler
import numpy as np

train_array = np.arange(0, 11).reshape(-1, 1)
print("\n train_array : ", train_array)

test_array = np.arange(0, 6).reshape(-1, 1)
print("\n test_array : ", test_array)

scaler = MinMaxScaler()
scaler.fit(train_array)
train_scaled = scaler.transform(train_array)
print("\n train_array : ", train_array.reshape(-1))
print("\n train_scaled : ", train_scaled.reshape(-1))

test_scaled = scaler.transform(test_array)
print("\n test_array : ", test_array.reshape(-1))
print("\n test_scaled : ", test_scaled.reshape(-1))