from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

print("LDA (Linear Discriminant Analysis)")

print("\n")

iris = load_iris()
iris_scaled = StandardScaler().fit_transform(iris.data)
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(iris_scaled, iris.target)
iris_lda = lda.transform(iris_scaled)

print("iris_lda.shape : ", iris_lda.shape)

df_lda = pd.DataFrame(iris_lda, columns=['component1', 'component2'])
df_lda['target'] = iris.target

markers = ['^', 's', 'o']

for i, marker in enumerate(markers):
    x_data = df_lda[df_lda['target'] == i]['component1']
    y_data = df_lda[df_lda['target'] == i]['component2']
    plt.scatter(x_data, y_data, marker=marker, label=iris.target_names[i])

plt.legend()
plt.xlabel('component1')
plt.ylabel('component2')
plt.show()