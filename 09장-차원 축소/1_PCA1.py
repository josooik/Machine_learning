from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

print("PCA(Principal Component Analysis)")

print("\n")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=['seal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['target'] = iris.target
print(df.head(3))

markers = ['^', 's', 'o']

for i, marker in enumerate(markers):
    x_data = df[df['target'] == i]['seal_length']
    y_data = df[df['target'] == i]['sepal_width']
    plt.scatter(x_data, y_data, marker=marker, label=iris.target_names[i])

plt.legend()
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

print("\n")

print("데이터 전처리")

print("\n")

iris_scaled = StandardScaler().fit_transform(df.iloc[:, :-1])

pca = PCA(n_components=2)
pca.fit(iris_scaled)
iris_pca = pca.transform(iris_scaled)
print("iris_pca.shape : ", iris_pca.shape)

df_pca = pd.DataFrame(iris_pca, columns=['component1', 'component2'])
df_pca['target'] = iris.target
print(df_pca.head(3))

markers = ['^', 's', 'o']

for i, marker in enumerate(markers):
    x_data = df_pca[df_pca['target'] == i]['component1']
    y_data = df_pca[df_pca['target'] == i]['component2']
    plt.scatter(x_data, y_data, marker=marker, label=iris.target_names[i])

plt.legend()
plt.xlabel('component1')
plt.ylabel('component2')
plt.show()

print("\n")

print("변동성 비율")
print("pca.explained_variance_ratio_ : ", pca.explained_variance_ratio_)

print("\n")

print("RandomForestClassifier")
rf = RandomForestClassifier()
scores = cross_val_score(rf, iris.data, iris.target, scoring='accuracy', cv=3)
print("scores : ", scores)
print("평균 scores : ", np.mean(scores))

print("\n")

print("PCA")
pca_X = df_pca[['component1', 'component2']]
scores = cross_val_score(rf, pca_X, iris.target, scoring='accuracy', cv=3)
print("scores : ", scores)
print("평균 scores : ", np.mean(scores))