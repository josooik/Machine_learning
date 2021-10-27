from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print("k-평균 알고리즘 (k-Means Algorithm)")

print("\n")

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
print("df.head(3) : \n", df.head(3))

print("\n")

kmeans = KMeans(n_clusters=3, max_iter=3000)
kmeans.fit(df)
print("kmeans.labels_ : \n", kmeans.labels_)

print("\n")

pca = PCA(n_components=2)
iris_pca = pca.fit_transform(iris.data)

df['cluster'] = kmeans.labels_
df['component1'] = iris_pca[:, 0]
df['component2'] = iris_pca[:, 1]
print("df.head(3) : \n", df.head(3))

marker0_index = df[df['cluster'] == 0].index
marker1_index = df[df['cluster'] == 1].index
marker2_index = df[df['cluster'] == 2].index

plt.scatter(x=df.loc[marker0_index, 'component1'], y=df.loc[marker0_index, 'component2'], marker='o')
plt.scatter(x=df.loc[marker1_index, 'component1'], y=df.loc[marker1_index, 'component2'], marker='s')
plt.scatter(x=df.loc[marker2_index, 'component1'], y=df.loc[marker2_index, 'component2'], marker='^')
plt.show()