from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

print("\n", iris_df.mean())
print("\n", iris_df.var())

scaler = StandardScaler()
scaler.fit(iris_df)
print("\n scaler : ", scaler)

iris_scaled = scaler.transform(iris_df)
print("\n iris_scaled : ", iris_scaled)

iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print("\n iris_df_scaled :", iris_df_scaled)
print("\n", iris_df_scaled.mean())
print("\n", iris_df_scaled.var())