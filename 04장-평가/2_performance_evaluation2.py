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


def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    return df

def drop_features(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df

def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

def get_clf_eval(y_test, predictions):
    print("오차 행렬(confusion_matrix) : \n", confusion_matrix(y_test, predictions))
    print("정확도(accuracy_score) : ", accuracy_score(y_test, predictions))
    print("정밀도(precision_score) : ", precision_score(y_test, predictions))
    print("재현율(recall_score) : ", recall_score(y_test, predictions))

titanic_df = pd.read_csv('./data/titanic/train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived', axis=1)
X_titanic_df = transform_features(X_titanic_df)

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2)

clf = LogisticRegression()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
get_clf_eval(y_test, pred)

print("\n")

pred_prob = clf.predict_proba(X_test)
print(pred_prob)

print("\n")

result = np.concatenate([pred_prob, pred.reshape(-1, 1)], axis=1)
print(result)

X = [[1, -1, 2],
     [2, 0, 0],
     [0, 1.1, 1.2]]

print("\n")

binarizer = Binarizer(threshold=1.1)
print(binarizer.fit_transform(X))

print("\n")

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11)

clf = LogisticRegression()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
pred_prob = clf.predict_proba(X_test)
pred_prob2 = pred_prob[:, 1].reshape(-1, 1)

th = 0.5
print('th = 0.5')
binarizer = Binarizer(threshold=th).fit(pred_prob2)
custom_pred = binarizer.transform(pred_prob2)
get_clf_eval(y_test, custom_pred)

print("\n")

th = 0.4
print('th = 0.4')
binarizer = Binarizer(threshold=th).fit(pred_prob2)
custom_pred = binarizer.transform(pred_prob2)
get_clf_eval(y_test, custom_pred)

print("\n")

ths = [0.4, 0.45, 0.5, 0.55, 0.6]
for th in ths:
    print('ths : ', th)
    binarizer = Binarizer(threshold=th).fit(pred_prob2)
    custom_pred = binarizer.transform(pred_prob2)
    get_clf_eval(y_test, custom_pred)
    print("\n")


pred_prob = clf.predict_proba(X_test)[:, 1]

precisions, recalls, ths = precision_recall_curve(y_test, pred_prob)
print("ths.shape : ", ths.shape)

th_index = np.arange(0, ths.shape[0], 15)
print("th_index : ", th_index)

print("ths[th_index] : ", ths[th_index])
print("precisions[th_index] : ", precisions[th_index])
print("recalls[th_index] : ", recalls[th_index])

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

def get_clf_eval(y_test, predictions):
    print("오차 행렬(confusion_matrix) : \n", confusion_matrix(y_test, predictions))
    print("정확도(accuracy_score) : ", accuracy_score(y_test, predictions))
    print("정밀도(precision_score) : ", precision_score(y_test, predictions))
    print("재현율(recall_score) : ", recall_score(y_test, predictions))
    print("F1(f1_score) : ", f1_score(y_test, predictions))

ths = [0.4, 0.45, 0.5, 0.55, 0.6]
for th in ths:
    print('ths : ', th)
    binarizer = Binarizer(threshold=th).fit(pred_prob2)
    custom_pred = binarizer.transform(pred_prob2)
    get_clf_eval(y_test, custom_pred)
    print("\n")

def roc_curve_plot(y_test, pred_prob):
    fprs, tprs, ths = roc_curve(y_test, pred_prob)
    plt.plot(fprs, tprs, label='ROC')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.show()

roc_curve_plot(y_test, pred_prob)

def get_clf_eval(y_test, predictions, pred_prob):
    print("오차 행렬(confusion_matrix) : \n", confusion_matrix(y_test, predictions))
    print("정확도(accuracy_score) : ", accuracy_score(y_test, predictions))
    print("정밀도(precision_score) : ", precision_score(y_test, predictions))
    print("재현율(recall_score) : ", recall_score(y_test, predictions))
    print("F1(f1_score) : ", f1_score(y_test, predictions))
    print("Roc(roc_auc_score) : ", roc_auc_score(y_test, pred_prob))

get_clf_eval(y_test, pred, pred_prob)