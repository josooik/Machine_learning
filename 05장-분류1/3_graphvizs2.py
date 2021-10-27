from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz
import pydot

# 결정트리
clf = DecisionTreeClassifier(min_samples_split=4)
iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
clf.fit(X_train, y_train)

with open("./data/tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

export_graphviz(clf, out_file="tree.dot", class_names=iris.target_names, feature_names=iris.feature_names, impurity=True, filled=True)

(graphviz,) = pydot.graph_from_dot_file('tree.dot', encoding='utf8')
graphviz.write_png("./결과파일/graphviz2.png")