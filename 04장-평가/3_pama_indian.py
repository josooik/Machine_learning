import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Binarizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

diabetes = pd.read_csv('./data/diabetes/diabetes.csv')
print(diabetes['Outcome'].value_counts())
print(diabetes.head(3))

print(diabetes.info())

X = diabetes.iloc[:, :-1]
y = diabetes.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156, stratify=y)

clf = LogisticRegression()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
pred_prob = clf.predict_proba(X_test)[:, 1]

def get_clf_eval(y_test, pred, pred_prob):
    print("오차 행렬(confusion_matrix) : \n", confusion_matrix(y_test, pred))
    print("정확도(accuracy_score) : ", accuracy_score(y_test, pred))
    print("정밀도(precision_score) : ", precision_score(y_test, pred))
    print("재현율(recall_score) : ", recall_score(y_test, pred))
    print("F1(f1_score) : ", f1_score(y_test, pred))
    print("Roc(roc_auc_score) : ", roc_auc_score(y_test, pred_prob))

get_clf_eval(y_test, pred, pred_prob)

def precision_recall_curve_plot(y_test, pred_prob):
    precisions, recalls, ths = precision_recall_curve(y_test, pred_prob)

    th_boundary = ths.shape[0]
    plt.plot(ths, precisions[0:th_boundary], linestyle='--', label='precision')
    plt.plot(ths, recalls[0:th_boundary], label='recall')

    plt.xlabel('thresholds')
    plt.legend()
    plt.grid()
    plt.show()

precision_recall_curve_plot(y_test, pred_prob)

print("\n")

print(diabetes.describe())

plt.hist(diabetes['Glucose'], bins=10)
plt.show()

print("\n")

zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', "BMI"]
total_count = diabetes['Glucose'].count()

for feature in zero_features:
    zero_count = diabetes[diabetes[feature] == 0][feature].count()
    print(feature, zero_count, 100 * zero_count / total_count)

print("\n")

mean_zero_features = diabetes[zero_features].mean()
diabetes[zero_features] = diabetes[zero_features].replace(0, mean_zero_features)

X = diabetes.iloc[:, :-1]
y = diabetes.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=156, stratify=y)

clf = LogisticRegression()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
pred_prob = clf.predict_proba(X_test)[:, 1]

def get_clf_eval(y_test, pred, pred_prob):
    print("오차 행렬(confusion_matrix) : \n", confusion_matrix(y_test, pred))
    print("정확도(accuracy_score) : ", accuracy_score(y_test, pred))
    print("정밀도(precision_score) : ", precision_score(y_test, pred))
    print("재현율(recall_score) : ", recall_score(y_test, pred))
    print("F1(f1_score) : ", f1_score(y_test, pred))
    print("Roc(roc_auc_score) : ", roc_auc_score(y_test, pred_prob))

get_clf_eval(y_test, pred, pred_prob)

print("\n")

def get_clf_eval(y_test, pred, pred_prob):
    print("오차 행렬(confusion_matrix) : \n", confusion_matrix(y_test, pred))
    print("정확도(accuracy_score) : ", accuracy_score(y_test, pred))
    print("정밀도(precision_score) : ", precision_score(y_test, pred))
    print("재현율(recall_score) : ", recall_score(y_test, pred))
    print("F1(f1_score) : ", f1_score(y_test, pred))
    print("Roc(roc_auc_score) : ", roc_auc_score(y_test, pred_prob))

pred = clf.predict(X_test)
pred_prob = clf.predict_proba(X_test)[:, 1].reshape(-1, 1)

ths = [0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.5]
for th in ths:
    print("th : ", th)
    binarizer = Binarizer(threshold=th).fit(pred_prob)
    custom_pred = binarizer.transform(pred_prob)
    get_clf_eval(y_test, pred, custom_pred)
    print("\n")

binarizer = Binarizer(threshold=0.48)

pred_48 = binarizer.fit_transform(pred_prob)
get_clf_eval(y_test, pred_48, pred_prob)