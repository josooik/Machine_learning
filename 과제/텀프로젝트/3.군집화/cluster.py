from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

x, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# MinMaxScaler 메소드로 전처리

scaler_MMS = MinMaxScaler().fit(x)

x_scaled_MMS = scaler_MMS.transform(x) # 전처리 메소드를 훈련데이터에 적용

dbscan = DBSCAN() # 모델생성

clusters_MMS = dbscan.fit_predict(x_scaled_MMS) # 모델 학습

print('np.unique(clusters_MMS)\n예측한 레이블: {}'.format(np.unique(clusters_MMS))) # [0]

# 예측한 레이블이 0으로 전부 하나의 클러스터로 표현

# MinMaxScaler전처리가 적합하지 않음

scaler_ss = StandardScaler().fit(x)
x_scaled_ss = scaler_ss.transform(x)
dbscan = DBSCAN()
clusters_ss = dbscan.fit_predict(x_scaled_ss)
print('np.unique(clusters_ss)\n예측한 레이블:{}'.format(np.unique(clusters_ss))) # [0 ,1]

# visualization

df = np.hstack([x_scaled_ss, clusters_ss.reshape(-1, 1)]) # x_scaled_ss 오른쪽에 1열 붙이기

df_ft0 = df[df[:,2]==0, :] # 클러스터 0 추출
df_ft1 = df[df[:,2]==1, :] # 클러스터 1 추출

plt.scatter(df_ft0[:, 0], df_ft0[:, 1], label='cluster 0', cmap='Pairs') # x, y, label, 색상
plt.scatter(df_ft1[:, 0], df_ft1[:, 1], label='cluster 1', cmap='Pairs')
plt.xlabel('feature 0')
plt.ylabel('feature 1')
plt.legend()
plt.show()