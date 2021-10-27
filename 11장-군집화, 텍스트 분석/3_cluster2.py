import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA



print("데이터 전처리 \n")

df = pd.read_excel('./data/retail.xlsx')


df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df = df[df['CustomerID'].notnull()]

df = df[df['Country'] == 'United Kingdom']

df['sale_amount'] = df['Quantity'] * df['UnitPrice']
df['CustomerID'] = df['CustomerID'].astype(int)

aggs = {
    'InvoiceDate' : 'max',
    'InvoiceNo' : 'count',
    'sale_amount' : 'sum'
}

cust_df = df.groupby('CustomerID').agg(aggs)
cust_df = cust_df.rename(columns={'InvoiceDate' : 'Recency', 'InvoiceNo' : 'Frequency', 'sale_amount' : 'Monetary'})
cust_df = cust_df.reset_index()


cust_df['Recency'] = dt.datetime(2011, 12, 10) - cust_df['Recency']
cust_df['Recency'] = cust_df['Recency'].apply(lambda x: x.days + 1)


print("데이터 분석 \n")


X = cust_df[['Recency', 'Frequency', 'Monetary']].values
x_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(x_scaled)
cust_df['label'] = labels

print("silhouette_score : ", silhouette_score(x_scaled, labels))

print("\n")

print("visualize_kmeans_plot_multi")

def visualize_kmeans_plot_multi(cluster_lists, X_features):

    # plt.subplots()으로 리스트에 기재된 클러스터링 만큼의 sub figures를 가지는 axs 생성
    n_cols = len(cluster_lists)
    fig, axs = plt.subplots(figsize=(4 * n_cols, 4), nrows=1, ncols=n_cols)
    # 입력 데이터의 FEATURE가 여러개일 경우 2차원 데이터 시각화가 어려우므로 PCA 변환하여 2차원 시각화
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(X_features)
    dataframe = pd.DataFrame(pca_transformed, columns=['PCA1', 'PCA2'])
    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 KMeans 클러스터링 수행하고 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        # KMeans 클러스터링으로 클러스터링 결과를 dataframe에 저장.
        clusterer = KMeans(n_clusters=n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(pca_transformed)
        dataframe['cluster'] = cluster_labels

        unique_labels = np.unique(clusterer.labels_)
        markers = ['o', 's', '^', 'x', '*']

        # 클러스터링 결과값 별로 scatter plot 으로 시각화
        for label in unique_labels:
            label_df = dataframe[dataframe['cluster'] == label]
            if label == -1:
                cluster_legend = 'Noise'
            else:
                cluster_legend = 'Cluster ' + str(label)
            axs[ind].scatter(x=label_df['PCA1'], y=label_df['PCA2'], s=70, \
                             edgecolor='k', marker=markers[label], label=cluster_legend)

        axs[ind].set_title('Number of Cluster : ' + str(n_cluster))
        axs[ind].legend(loc='upper right')

    plt.show()

visualize_kmeans_plot_multi([2, 3, 4, 5], x_scaled)

print("\n")

print("왜곡 완화 -> log 변환")

print("\n")

cust_df['Recency_log'] = np.log1p(cust_df['Recency'])
cust_df['Frequency_log'] = np.log1p(cust_df['Frequency'])
cust_df['Monetary_log'] = np.log1p(cust_df['Monetary'])

X = cust_df[['Recency_log', 'Frequency_log', 'Monetary_log']].values
x_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(x_scaled)
cust_df['label'] = labels

print("silhouette_score : ", silhouette_score(x_scaled, labels))

visualize_kmeans_plot_multi([2, 3, 4, 5], x_scaled)

