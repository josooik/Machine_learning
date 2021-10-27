from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=121)
dtree = DecisionTreeClassifier()
parameters = {'max_depth' : [1, 2, 3], 'min_samples_split' : [2, 3]}

grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv=3, refit=True)
grid_dtree.fit(X_train, y_train)

socres_df = pd.DataFrame(grid_dtree.cv_results_)
print(socres_df[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score']])
print(grid_dtree.best_params_)
print(grid_dtree.best_score_)

estimator = grid_dtree.best_estimator_
pred = estimator.predict(X_test)
print("estimator : ", accuracy_score(y_test, pred))