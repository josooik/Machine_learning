from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("LDA (Linear Discriminant Analysis)")

print("\n")

iris = load_iris()
iris_data = iris.data
tsvd = TruncatedSVD(n_components=2)
tsvd.fit(iris_data)
iris_tsvd = tsvd.transform(iris_data)

plt.scatter(x=iris_tsvd[:, 0], y = iris_tsvd[:, 1], c=iris.target)
plt.show()

iris_scaled = StandardScaler().fit_transform(iris_data)

tsvd = TruncatedSVD(n_components=2)
tsvd.fit(iris_scaled)
iris_tsvd = tsvd.transform(iris_scaled)

pca = PCA(n_components=2)
pca.fit(iris_scaled)
iris_pca = pca.transform(iris_scaled)

fig, (ax1, ax2) = plt.subplots(figsize=(9, 4), ncols=2)
ax1.scatter(x=iris_tsvd[:, 0], y=iris_tsvd[:, 1], c=iris.target)
ax2.scatter(x=iris_tsvd[:, 0], y=iris_pca[:, 1], c=iris.target)
ax1.set_title('Truncated SVD')
ax2.set_title('PCA')
plt.show()

print("\n")

print("평균 (iris_pca - iris_tsvd) : ", (iris_pca - iris_tsvd).mean())
print("평균 (pca.components_ - tsvd.components_) : ", (pca.components_ - tsvd.components_).mean())