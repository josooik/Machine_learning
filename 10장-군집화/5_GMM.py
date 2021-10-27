import pandas as pd
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

print("GMM (Gaussian Mixture Model) \n")

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['target'] = iris.target

gmm = GaussianMixture(n_components=3).fit(iris.data)
gmm_labels = gmm.predict(iris.data)
df['gmm_cluster'] = gmm_labels

iris_result = df.groupby(['target'])['gmm_cluster'].value_counts()
print("iris_result :", iris_result)

print("\n")

kmeans = KMeans(n_clusters=3, max_iter=300).fit(iris.data)
kmeans_labels = kmeans.predict(iris.data)
df['kmeans_cluster'] = kmeans_labels

iris_result = df.groupby(['target'])['kmeans_cluster'].value_counts()
print("iris_result :", iris_result)