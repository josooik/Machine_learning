import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv('./data/hw.csv')
print(df.head(3))
print(df.info())

X = df[['f1', 'f2', 'f3', 'f4', 'f5']].values
x_scaled = StandardScaler().fit_transform(X)
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(x_scaled, df.target)
df_lda = lda.transform(x_scaled)
print(df_lda.shape)

df_lda = pd.DataFrame(df_lda, columns=['component1', 'component2'])
df_lda['target'] = df.target

markers = ['^', 's', 'o']

for i, marker in enumerate(markers):
    x_data = df_lda[df_lda['target'] == i]['component1']
    y_data = df_lda[df_lda['target'] == i]['component2']
    plt.scatter(x_data, y_data, marker=marker)

plt.xlabel('component1')
plt.ylabel('component2')
plt.show()

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

        plt.scatter(x=label_cluster['component1'], y=label_cluster['component2'], s=70, \
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


kmeans = KMeans(n_clusters=2)
kmeans_label = kmeans.fit_predict(x_scaled)
df_lda['kmeans_label'] = kmeans_label
visualize_cluster_plot(kmeans, df_lda, 'kmeans_label', iscenter=True)
print("silhouette_score : ", silhouette_score(x_scaled, kmeans_label))