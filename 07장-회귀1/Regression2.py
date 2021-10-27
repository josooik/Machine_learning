from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

print("다항 회귀 과적합 \n")

def true_fun(X):
    return np.cos(1.5 * np.pi * X)

np.random.seed(0)
n_samples = 30
X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1
plt.figure(figsize=(14, 5))
degrees = [1, 4, 15]

X_mat = X.reshape(-1, 1)
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i+1)
    plt.setp(ax, xticks=(), yticks=())

    poly = PolynomialFeatures(degree=degrees[i], include_bias=False)
    lr = LinearRegression()
    pipeline = Pipeline([("polynomial_features", poly), ("linear_regression", lr)])
    pipeline.fit(X_mat, y)

    scores = cross_val_score(pipeline, X_mat, y, scoring="neg_mean_squared_error", cv=10)
    coefficients = pipeline.named_steps['linear_regression'].coef_
    print(degrees[i], coefficients)
    print(degrees[i], -1 * np.mean(scores))

    X_test = np.linspace(0, 1, 100)
    pred = pipeline.predict(X_test.reshape(-1, 1))
    plt.plot(X_test, pred, label="Model")
    plt.plot(X_test, true_fun(X_test), '-', label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.xlim(0, 1)
    plt.ylim(-2, 2)
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(degrees[i], -scores.mean(), scores.std()))

plt.show()