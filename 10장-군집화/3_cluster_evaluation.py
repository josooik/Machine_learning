from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
kmeans = KMeans(n_clusters=3, max_iter=300).fit(df)
df['cluster'] = kmeans.labels_

score_samples = silhouette_samples(iris.data, df['cluster'])
print("score_samples.shape : ", score_samples.shape)

print("\n")

df['silhouette_coeff'] = score_samples
average_score = silhouette_score(iris.data, df['cluster'])
print("average_score : ", average_score)

print("\n")

print(df.head(3))

print("\n")

print(df.groupby('cluster')['silhouette_coeff'].mean())