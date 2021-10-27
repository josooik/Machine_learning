import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 6 + 4 * X + np.random.randn(100, 1)

def get_cost(y, y_pred):
    N = len(y)
    cost = np.sum(np.square(y - y_pred)) / N
    return cost

def get_weight_updates(w1, w0, X, y, learning_rate=0.01):
    N = len(y)
    y_pred = np.dot(X, w1) + w0
    #print("get_cost(y, y_pred) : ", get_cost(y, y_pred))
    diff = y_pred - y
    ones = np.ones((N, 1))
    w1_update = learning_rate * 2 * np.dot(X.T, diff) / N
    w0_update = learning_rate * 2 * np.dot(ones.T, diff) / N
    return w1_update, w0_update

def gradient_descent_steps(X, y, iters=10000):
    w0 = 0
    w1 = 0
    for _ in range(iters):
        w1_update, w0_update = get_weight_updates(w1, w0, X, y)
        w1 = w1 - w1_update
        w0 = w0 - w0_update
    return w1, w0

w1, w0 = gradient_descent_steps(X, y, 1000)
print("w1, w0 : ", w1, w0)

y_pred = w1 * X + w0
plt.scatter(X, y)
plt.plot(X, y_pred, c='red')
plt.show()