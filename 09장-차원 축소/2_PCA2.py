import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

print("PCA(Principal Component Analysis)")

print("\n")

df = pd.read_excel('./data/credit_card.xls', header=1, sheet_name='Data').iloc[0:, 1:]
print("df.shape : ", df.shape)
print("df.head(3) \n", df.head(3))

df.rename(columns={'PAY_0':'PAY_1', 'default payment next month':'default'}, inplace=True)
y_target = df['default']
x_features = df.drop('default', axis=1)

corr = x_features.corr()
plt.figure(figsize=(14, 14))
sns.heatmap(corr, annot=True, fmt='.1g')
plt.show()

print("\n")

cols_bill = ['BILL_AMT' + str(i) for i in range(1, 7)]
print("cols_bill : ", cols_bill)

df_scaled = StandardScaler().fit_transform(x_features[cols_bill])
pca = PCA(n_components=2)
pca.fit(df_scaled)
print("pca.explained_variance_ratio_ : ", pca.explained_variance_ratio_)

print("\n")

print("RandomForestClassifier")
rf = RandomForestClassifier(n_estimators=300)
scores = cross_val_score(rf, x_features, y_target, scoring='accuracy', cv=3)
print("scores : ", scores)
print("평균 scores : ", np.mean(scores))

print("\n")

print("PCA")
df_scaled = StandardScaler().fit_transform(x_features)
pca = PCA(n_components=6)
df_pca = pca.fit_transform(df_scaled)
scores = cross_val_score(rf, df_pca, y_target, scoring='accuracy', cv=3)
print("scores : ", scores)
print("평균 scores : ", np.mean(scores))