from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# 결정 트리 과적합(Overfitting)
X_features, y_labels = make_classification(n_features=2, n_redundant=0, n_classes=3, n_clusters_per_class=1)
plt.scatter(X_features[:, 0], X_features[:, 1], marker='o', c=y_labels, s=25, edgecolor='k')
plt.show()