from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 피처의 중요도
clf = DecisionTreeClassifier()
iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
clf.fit(X_train, y_train)

print("feature_names : ", iris.feature_names)
print("feature_importances : ", clf.feature_importances_)

sns.barplot(x=clf.feature_importances_, y=iris.feature_names)
plt.show()