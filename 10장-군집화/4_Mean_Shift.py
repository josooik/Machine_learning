import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth

print("평균 이동(Mean Shift) \n")

X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.7)
meanshift = MeanShift(bandwidth=0.8)
cluster_labels = meanshift.fit_predict(X)
print(np.unique(cluster_labels))

print("\n")

meanshift = MeanShift(bandwidth=1)
cluster_labels = meanshift.fit_predict(X)
print(np.unique(cluster_labels))

print("\n")

bandwidth = estimate_bandwidth(X)
print("bandwidth : ", bandwidth)

print("\n")

meanshift = MeanShift(bandwidth=bandwidth)
cluster_labels = meanshift.fit_predict(X)
print(np.unique(cluster_labels))

print("\n")

df = pd.DataFrame(data=X, columns=['feature1', 'feature2'])
df['target'] = y

df['meanshift_label'] = cluster_labels
centers = meanshift.cluster_centers_
unique_labels = np.unique(cluster_labels)
markers = ['o', 's', '^', 'x', '*']

for label in unique_labels:
    label_cluster = df[df['meanshift_label'] == label]
    center = centers[label]
    plt.scatter(x=label_cluster['feature1'], y=label_cluster['feature2'],
                edgecolors='k', marker=markers[label])
    plt.scatter(x=center[0], y=center[1], s=200, color='gray', alpha=0.9, marker=markers[label])
    plt.scatter(x=center[0], y=center[1], s=70, color='k', edgecolors='k', marker='$%d$' % label)

plt.show()