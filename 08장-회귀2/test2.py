from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

cancer = load_breast_cancer()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(cancer.data)

X_train, X_test, y_train, y_test = train_test_split(data_scaled, cancer.target, test_size=0.3)

print("로지스틱(Logistic) 회귀 \n")

lr = LogisticRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

print("accuracy_score : ", accuracy_score(y_test, pred))
print("roc_auc_score : ", roc_auc_score(y_test, pred))

print("\n")

params = {
    'penalty' : ['l2', 'l1'],
    'C' : [0.01, 0.1, 1, 5, 10],
    'solver' : ['liblinear']
}

grid_clf = GridSearchCV(lr, param_grid=params, scoring='accuracy', cv=3)
grid_clf.fit(data_scaled, cancer.target)
print("grid_clf.best_params : %s, grid_clf.best_score : %f" %(grid_clf.best_params_, grid_clf.best_score_))
