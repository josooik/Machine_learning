import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

print("GMM vs K-Means \n")

def visualize_cluster_plot(clusterobj, dataframe, label_name, iscenter=True):
    if iscenter:
        centers = clusterobj.cluster_centers_

    unique_labels = np.unique(dataframe[label_name].values)
    markers = ['o', 's', '^', 'x', '*']
    isNoise = False

    for label in unique_labels:
        label_cluster = dataframe[dataframe[label_name] == label]
        if label == -1:
            cluster_legend = 'Noise'
            isNoise = True
        else:
            cluster_legend = 'Cluster ' + str(label)

        plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], s=70, \
                    edgecolor='k', marker=markers[label], label=cluster_legend)

        if iscenter:
            center_x_y = centers[label]
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=250, color='white',
                        alpha=0.9, edgecolor='k', marker=markers[label])
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k', \
                        edgecolor='k', marker='$%d$' % label)
    if isNoise:
        legend_loc = 'upper center'
    else:
        legend_loc = 'upper right'

    plt.legend(loc=legend_loc)
    plt.show()

X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=0.5)
transformation = [[0.6083, -0.6367],
                  [-0.4089, 0.8525]]

X2 = np.dot(X, transformation)
df = pd.DataFrame(data=X2, columns=['ftr1', 'ftr2'])
df['target'] = y
visualize_cluster_plot(None, df, 'target', iscenter=False)

print("K-Means \n")
kmeans = KMeans(3)
kmeans_label = kmeans.fit_predict(X2)
df['kmeans_label'] = kmeans_label
visualize_cluster_plot(kmeans, df, 'kmeans_label', iscenter=True)

print("GMM \n")
gmm = GaussianMixture(n_components=3)
gmm_label = gmm.fit_predict(X2)
df['gmm_label'] = gmm_label
visualize_cluster_plot(gmm, df, 'gmm_label', iscenter=False)

print(df.groupby('target')['kmeans_label'].value_counts())
print("\n")
print(df.groupby('target')['gmm_label'].value_counts())