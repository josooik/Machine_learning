import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
import pydot
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier(min_samples_leaf=5)
cancer = load_breast_cancer()

df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)

with open("breast_cancer.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

export_graphviz(clf, out_file="breast_cancer.dot", class_names=cancer.target_names, feature_names=cancer.feature_names, impurity=True, filled=True)

(graphviz,) = pydot.graph_from_dot_file('./breast_cancer.dot', encoding='utf8')
graphviz.write_png("./결과파일/result(min_samples_leaf=5).png")

print("accuracy : ", accuracy)