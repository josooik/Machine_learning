import numpy as np
import matplotlib.pyplot as plt

print("중심점 최적화")

def get_min_dist_cluster(pt, centers, n_clusters):
    return np.argmin(np.sqrt(np.sum((pt - centers) ** 2, axis=1)))

def init_centers(points, n_clusters):
    n_samples, n_features = points.shape
    centers = np.empty((n_clusters, n_features), dtype=points.dtype)
    centers[0] = points[np.random.choice(n_samples, 1)]
    for i in range(1, n_clusters):
        centers[i] = points[np.argmax([np.min(np.sqrt(np.sum((pt - centers) ** 2, axis=1))) for pt in points])]
    return centers

def find_clusters(points, n_clusters):
    centers = init_centers(points, n_clusters)

    for _ in range(10):
        labels = np.array([get_min_dist_cluster(pt, centers, n_clusters) for pt in points])
        centers = np.array([np.mean(points[labels == i], axis=0) for i in range(n_clusters)])
    return centers, labels

points = np.genfromtxt('./data/data.csv', delimiter=',', skip_header=1)
num_clusters = 3

centers, labels = find_clusters(points, num_clusters)
cmap = plt.cm.get_cmap('viridis', num_clusters)
plt.scatter(points[:,0], points[:,1], c=labels, cmap=cmap)
plt.scatter(centers[:,0], centers[:,1], c='red')

plt.xlim(0, 20)
plt.ylim(0, 20)
plt.show()
