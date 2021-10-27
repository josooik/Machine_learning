import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
import numpy as np


df = pd.read_csv('./data/hw.csv')
print(df.head(3))
print(df.info())

X = df[['f1', 'f2', 'f3', 'f4', 'f5']].values
x_scaled = StandardScaler().fit_transform(X)
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(x_scaled, df.target)
df_lda = lda.transform(x_scaled)
print(df_lda.shape)

df_lda = pd.DataFrame(df_lda, columns=['feature1', 'feature2'])
df_lda['target'] = df.target

markers = ['^', 's', 'o']

for i, marker in enumerate(markers):
    x_data = df_lda[df_lda['target'] == i]['feature1']
    y_data = df_lda[df_lda['target'] == i]['feature2']
    plt.scatter(x_data, y_data, marker=marker)

plt.xlabel('feature1')
plt.ylabel('feature2')

bandwidth = estimate_bandwidth(x_scaled)
print(bandwidth)

meanshift = MeanShift(bandwidth=bandwidth)
cluster_labels = meanshift.fit_predict(x_scaled)
print(np.unique(cluster_labels))

df_lda['meanshift_label'] = cluster_labels
centers = meanshift.cluster_centers_
unique_labels = np.unique(cluster_labels)
makers = ['o', 's', '^', 'x', '*']

for label in unique_labels:
    label_cluster = df_lda[df_lda['meanshift_label'] == label]
    center = centers[label]
    plt.scatter(x=label_cluster['feature1'], y=label_cluster['feature2'],
                edgecolor='k')
    #plt.scatter(x=center[0], y=center[1], s=200, color='gray', alpha=0.9)
    #plt.scatter(x=center[0], y=center[1], s=70, color='k', edgecolor='k')

plt.show()