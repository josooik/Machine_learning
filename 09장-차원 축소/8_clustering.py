from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

print("군집화(clustering)")
print("군집화 데이터 생성")

print("\n")

X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.8)
print("X.shape : %s, y.shape : %s" % (X.shape, y.shape))

print("\n")

unique, counts = np.unique(y, return_counts=True)
print("unique : %s, counts : %s" % (unique, counts))

print("\n")

df = pd.DataFrame(data=X, columns=['feature1', 'feature2'])
df['target'] = y
print("df.head(3) : \n", df.head(3))

targets = np.unique(y)
markers = ['o', 's', '^']
for target in targets:
    cluster = df[df['target'] == target]
    plt.scatter(x=cluster['feature1'], y=cluster['feature2'], edgecolor='k', marker=markers[target])

plt.show()

kmeans = KMeans(n_clusters=3, max_iter=200)
labels = kmeans.fit_predict(X)
df['label'] = labels
centers = kmeans.cluster_centers_
uniques = np.unique(labels)
markers = ['o', 's', '^']

for label in uniques:
    cluster = df[df['label'] == label]
    center = centers[label]
    plt.scatter(x=cluster['feature1'], y=cluster['feature2'], edgecolor='k', marker=markers[label])
    plt.scatter(x=center[0], y=center[1], s=200, color='white', alpha=0.9, edgecolor='k', marker=markers[label])
    plt.scatter(x=center[0], y=center[1], s=70, color='k', edgecolor='k', marker='$%d$' % label)

plt.show()