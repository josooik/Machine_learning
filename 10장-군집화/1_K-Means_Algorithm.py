import numpy as np
import matplotlib.pyplot as plt

print("판다스(Pandas) 없이 구현하기1 \n")

def idstance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return np.sqrt(dx**2 + dy**2)

def get_min_dist_cluster(pt, centers, n_clusters):
    min_index = 0
    min_dist = float('inf')
    for i in range(n_clusters):
        dist = distance(pt, centers[i])
        if dist < min_dist:
            min_dist = dist
            min_index = i

    return min_index

print("판다스(Pandas) 없이 구현하기2 \n")

def idstance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def get_min_dist_cluster(pt, centers, n_clusters):
    min_index = 0
    min_dist = float('inf')
    for i in range(n_clusters):
        dist = distance(pt, centers[i])
        if dist < min_dist:
            min_dist = dist
            min_index = i

    return min_index

print("판다스(Pandas) 없이 구현하기3 \n")

def get_min_dist_cluster(pt, centers, n_clusters):
    return np.argmin(np.sqrt(np.sum((pt - centers) ** 2, axis=1)))

print("판다스(Pandas) 없이 구현하기1 \n")

def find_clusters(points, n_clusters):
    N = points.shape[0]
    indices = np.random.choice(N, n_clusters)
    centers = []
    for i in indices: centers.append(points[i])
    centers = np.array(centers)
    labels = [0] * N
    for _ in range(10):
        for i in range(N):
            labels[i] = get_min_dist_cluster(points[i], centers, n_clusters)
        for i in range(n_clusters):
            sums = [0, 0]
            count = 0
            for j in range(N):
                if labels[j] == i:
                    sums += points[j]
                    count += 1
            centers[i] = sums / count
    return centers, labels

points = np.genfromtxt('./data/data.csv', delimiter=',', skip_header=1)
num_clusters = 2

centers, labels = find_clusters(points, num_clusters)
cmap = plt.cm.get_cmap('viridis', num_clusters)
plt.scatter(points[:,0], points[:,1], c=labels, cmap=cmap)
plt.scatter(centers[:,0], centers[:,1], c='red')
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.show()

print("판다스(Pandas) 없이 구현하기2 \n")

def find_clusters(points, n_clusters):
    N = points.shape[0]
    centers = points[np.random.choice(N, n_clusters)]
    labels = np.array([0] * N)
    for _ in range(10):
        for i in range(N):
            labels[i] = get_min_dist_cluster(points[i], centers, n_clusters)
        for i in range(n_clusters):
            centers[i] = np.mean(points[labels == i], axis=0)
    return centers, labels

points = np.genfromtxt('./data/data.csv', delimiter=',', skip_header=1)
num_clusters = 2

centers, labels = find_clusters(points, num_clusters)
cmap = plt.cm.get_cmap('viridis', num_clusters)
plt.scatter(points[:,0], points[:,1], c=labels, cmap=cmap)
plt.scatter(centers[:,0], centers[:,1], c='red')

plt.xlim(0, 20)
plt.ylim(0, 20)
plt.show()

print("판다스(Pandas) 없이 구현하기3 \n")

def find_clusters(points, n_clusters):
    N = points.shape[0]
    centers = points[np.random.choice(N, n_clusters)]

    for _ in range(10):
        labels = np.array([get_min_dist_cluster(pt, centers, n_clusters) for pt in points])
        centers = np.array([np.mean(points[labels == i], axis=0) for i in range(n_clusters)])
    return centers, labels


points = np.genfromtxt('./data/data.csv', delimiter=',', skip_header=1)
num_clusters = 2

centers, labels = find_clusters(points, num_clusters)
cmap = plt.cm.get_cmap('viridis', num_clusters)
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap=cmap)
plt.scatter(centers[:, 0], centers[:, 1], c='red')

plt.xlim(0, 20)
plt.ylim(0, 20)
plt.show()

print("판다스 없이 구현하기 전체 코드1 \n")

import numpy as np
import matplotlib.pyplot as plt

def get_min_dist_cluster(pt, centers, n_clusters):
    return np.argmin(np.sqrt(np.sum((pt - centers) ** 2, axis=1)))


def find_clusters(points, n_clusters):
    N = points.shape[0]
    centers = points[np.random.choice(N, n_clusters)]

    for _ in range(10):
        labels = np.array([get_min_dist_cluster(pt, centers, n_clusters) for pt in points])
        centers = np.array([np.mean(points[labels == i], axis=0) for i in range(n_clusters)])
    return centers, labels


points = np.genfromtxt('./data/data.csv', delimiter=',', skip_header=1)
num_clusters = 2

centers, labels = find_clusters(points, num_clusters)
cmap = plt.cm.get_cmap('viridis', num_clusters)
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap=cmap)
plt.scatter(centers[:, 0], centers[:, 1], c='red')

plt.xlim(0, 20)
plt.ylim(0, 20)
plt.show()

print("판다스 없이 구현하기 전체 코드2 \n")

import numpy as np
import matplotlib.pyplot as plt

def get_min_dist_cluster(pt, centers, n_clusters):
    return np.argmin(np.sqrt(np.sum((pt - centers) ** 2, axis=1)))


def find_clusters(points, n_clusters):
    N = points.shape[0]
    centers = points[np.random.choice(N, n_clusters)]

    for _ in range(10):
        labels = np.array([get_min_dist_cluster(pt, centers, n_clusters) for pt in points])
        centers = np.array([np.mean(points[labels == i], axis=0) for i in range(n_clusters)])
    return centers, labels


points = np.genfromtxt('./data/data.csv', delimiter=',', skip_header=1)
num_clusters = 3

centers, labels = find_clusters(points, num_clusters)
cmap = plt.cm.get_cmap('viridis', num_clusters)
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap=cmap)
plt.scatter(centers[:, 0], centers[:, 1], c='red')

plt.xlim(0, 20)
plt.ylim(0, 20)
plt.show()