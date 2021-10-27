from sklearn.decomposition import NMF
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

print("NMF (Non-Negative Matrix Factorization)")

print("\n")

iris = load_iris()
iris_data = iris.data
nmf = NMF(n_components=2)
nmf.fit(iris_data)
iris_nmf = nmf.transform(iris_data)
plt.scatter(x=iris_nmf[:, 0], y=iris_nmf[:, 1], c=iris.target)
plt.show()