import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

print("스태킹 (Stacking) 앙상블 \n")
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=4)
rf = RandomForestClassifier(n_estimators=100)
dt = DecisionTreeClassifier()
ada = AdaBoostClassifier(n_estimators=100)
lr = LogisticRegression(C=10)

knn.fit(X_train, y_train)
rf.fit(X_train, y_train)
dt.fit(X_train, y_train)
ada.fit(X_train, y_train)

knn_pred = knn.predict(X_test)
rf_pred = rf.predict(X_test)
dt_pred = dt.predict(X_test)
ada_pred = ada.predict(X_test)

print("knn_pred_accuracy_score : ", accuracy_score(y_test, knn_pred))
print("rf_pred_accuracy_score : ", accuracy_score(y_test, rf_pred))
print("dt_pred_accuracy_score : ", accuracy_score(y_test, dt_pred))
print("ada_pred_accuracy_score : ", accuracy_score(y_test, ada_pred))

print("\n")

pred = np.array([knn_pred, rf_pred, dt_pred, ada_pred])
print("pred_shape : ", pred.shape)

pred = np.transpose(pred)
print("pred_shape : ", pred.shape)

lr.fit(pred, y_test)
pred_final = lr.predict(pred)

print("pred_final_accuracy_score : ", accuracy_score(y_test, pred_final))

print("\n")

print("CV 세트 기반의 스태킹 \n")
def get_stacking_base_datasets(model, X_train, y_train, X_test, n_folds):
    kf = KFold(n_splits=n_folds, shuffle=False)
    train_pred = np.zeros((X_train.shape[0], 1))
    test_pred = np.zeros((X_test.shape[0], n_folds))

    for folder_index, (train_index, valid_indx) in enumerate(kf.split(X_train)):
        X_tr = X_train[train_index]
        y_tr = y_train[train_index]
        X_te = X_train[valid_indx]
        model.fit(X_tr, y_tr)
        train_pred[valid_indx, :] = model.predict(X_te).reshape(-1, 1)
        test_pred[:, folder_index] = model.predict(X_test)

    test_mean = np.mean(test_pred, axis=1).reshape(-1, 1)
    return train_pred, test_mean

knn_train, knn_test = get_stacking_base_datasets(knn, X_train, y_train, X_test, 7)
rf_train, rf_test = get_stacking_base_datasets(rf, X_train, y_train, X_test, 7)
dt_train, dt_test = get_stacking_base_datasets(dt, X_train, y_train, X_test, 7)
ada_train, ada_test = get_stacking_base_datasets(ada, X_train, y_train, X_test, 7)

X_train_final = np.concatenate((knn_train, rf_train, dt_train, ada_train), axis=1)
X_test_final = np.concatenate((knn_test, rf_test, dt_test, ada_test), axis=1)
print("X_train.shape, X_test.shape : ", X_train.shape, X_test.shape)
print("X_train_final.shape, X_test_final.shape : ", X_train_final.shape, X_test_final.shape)

lr.fit(X_train_final, y_train)
pred_final = lr.predict(X_test_final)
print("accuracy_score : ", accuracy_score(y_test, pred_final))