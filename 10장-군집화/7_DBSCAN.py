import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.datasets import make_circles
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth

print("DBSCAN \n")

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

print("사이킷-런 DBSCAN 클래스 파라미터 \n")
# eps : 입실론 반경
# min_samples : 최소 데이터 개수(자신 포함)

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'peta_width'])
df['target'] = iris.target

dbscan = DBSCAN(eps=0.6, min_samples=8)
dbscan_label = dbscan.fit_predict(iris.data)
df['dbscan_label'] = dbscan_label

print(df.groupby('target')['dbscan_label'].value_counts())    # -1은 잡음(노이즈) 군집을 의미함

print("\n")

print("PCA \n")
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(iris.data)
df['ftr1'] = pca_transformed[:, 0]
df['ftr2'] = pca_transformed[:, 1]
visualize_cluster_plot(dbscan, df, 'dbscan_label', iscenter=False)

print("eps를 크게 하면 반경이 커져서 잡음 데이터 감소 \n")
dbscan = DBSCAN(eps=0.8, min_samples=8)
dbscan_label = dbscan.fit_predict(iris.data)
df['dbscan_label'] = dbscan_label

print(df.groupby('target')['dbscan_label'].value_counts())
df['ftr1'] = pca_transformed[:, 0]
df['ftr2'] = pca_transformed[:, 1]
visualize_cluster_plot(dbscan, df, 'dbscan_label', iscenter=False)

print("\n")

print("eps은 원래대로 0.6, min_samples를 16으로 변경  잡음 포인트 증가 \n")
dbscan = DBSCAN(eps=0.6, min_samples=16)
dbscan_label = dbscan.fit_predict(iris.data)
df['dbscan_label'] = dbscan_label

print(df.groupby('target')['dbscan_label'].value_counts())
df['ftr1'] = pca_transformed[:, 0]
df['ftr2'] = pca_transformed[:, 1]
visualize_cluster_plot(dbscan, df, 'dbscan_label', iscenter=False)

print("\n")

print("make_circles() 파라미터 \n")
# noise : 잡음 데이터 비율
# factor: 외부 원과 내부 원의 스케일 비율
X, y = make_circles(n_samples=1000, shuffle=True, noise=0.05, factor=0.5)
df = pd.DataFrame(data=X, columns=['ftr1', 'ftr2'])
df['target'] = y
visualize_cluster_plot(None, df, 'target', iscenter=False)

print("K-Means \n")
kmeans = KMeans(n_clusters=2, max_iter=1000)
kmeans_label = kmeans.fit_predict(X)
df['kmeans_label'] = kmeans_label
visualize_cluster_plot(kmeans, df, 'kmeans_label', iscenter=True)

print("Mean_Shift \n")
bandwidth = estimate_bandwidth(X)
meanshift = MeanShift(bandwidth=bandwidth)
meanshift_label = meanshift.fit_predict(X)
df['meanshift_label'] = meanshift_label
visualize_cluster_plot(meanshift, df, 'meanshift_label', iscenter=True)

print("GMM \n")
gmm = GaussianMixture(n_components=2)
gmm_label = gmm.fit_predict(X)
df['gmm_label'] = gmm_label
visualize_cluster_plot(gmm, df, 'gmm_label', iscenter=False)

print("DBSCAN \n")
dbscan = DBSCAN(eps=0.2, min_samples=10)
dbscan_label = dbscan.fit_predict(X)
df['dbscan_label'] = dbscan_label
visualize_cluster_plot(dbscan, df, 'dbscan_label', iscenter=False)