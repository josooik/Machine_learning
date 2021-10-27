from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

cancer = load_breast_cancer()

df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target
cancer_scaled = StandardScaler().fit_transform(cancer.data)

pca = PCA(n_components=2)
pca.fit(cancer_scaled)
cancer_pca = pca.transform(cancer_scaled)
print("cancer_pca.shape : ", cancer_pca.shape)

df_pca = pd.DataFrame(cancer_pca, columns=['component1', 'component2'])
df_pca['target'] = cancer.target
print(df_pca.head(3))

markers = ['^', 's']

for i, marker in enumerate(markers):
    x_data = df_pca[df_pca['target'] == i]['component1']
    y_data = df_pca[df_pca['target'] == i]['component2']
    plt.scatter(x_data, y_data, marker=marker, label=cancer.target_names[i])

plt.legend()
plt.title('PCA')
plt.xlabel('component1')
plt.ylabel('component2')
plt.show()