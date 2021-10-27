import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from lightgbm import LGBMClassifier
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve

df = pd.read_csv("./data/Ex06/creditcard.csv")
print(df.head(3))

# 신용 카드 사기 검출 : 전처리
def get_preprocessed_df(df):
    df_copy = df.copy()
    df_copy.drop('Time', axis=1, inplace=True)
    return df_copy

def get_train_test_dataset(df):
    df_copy = get_preprocessed_df(df)
    X_features = df_copy.iloc[:, :-1]
    y_target = df_copy.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3)
    return X_train, X_test, y_train, y_test

def get_clf_eval(y_test, pred=None, pred_prob=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_prob)

    print("confusion : ", confusion)
    print("accuracy : ", accuracy)
    print("precision : ", precision)
    print("recall : ", recall)
    print("f1 : ", f1)
    print("roc_auc : ", roc_auc)

def get_model_train_eval(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred_prob = model.predict_proba(X_test)[:, 1]
    get_clf_eval(y_test, pred, pred_prob)

def get_preprocessed_df(df):
    df_copy = df.copy()
    scaler = StandardScaler()
    amount_n = scaler.fit_transform(df_copy['Amount'].values.reshape(-1, 1))
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    df_copy.drop(['Time', 'Amount'], axis=1, inplace=True)
    return df_copy


X_train, X_test, y_train, y_test = get_train_test_dataset(df)
print(y_train.value_counts() / y_train.shape[0] * 100)
print(y_test.value_counts() / y_test.shape[0] * 100)

print("\n")

print("Logistic Regression \n")

lr = LogisticRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
pred_prob = lr.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, pred, pred_prob)

print("\n")

print("LightGBM \n")
lgbm = LGBMClassifier(n_estimators=1000, rum_levels=64, n_jobs=-1, boost_from_average=False)
get_model_train_eval(lgbm, X_train, X_test, y_train, y_test)

plt.figure(figsize=(8, 4))
plt.xticks(range(0, 30000, 1000), rotation=60)
sns.displot(df['Amount'])
plt.show()

print("\n")

print("Amount를 정규 분포로 변환 \n")
X_train, X_test, y_train, y_test = get_train_test_dataset(df)
lr = LogisticRegression()
get_model_train_eval(lr, X_train, X_test, y_train, y_test)

print("\n")
lgbm = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=-1)
get_model_train_eval(lgbm, X_train, X_test, y_train, y_test)

print("\n")

print("Log 변환 \n")
def get_preprocessed_df(df):
    df_copy = df.copy()
    amount_n = np.log1p(df_copy["Amount"])
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    df_copy.drop(['Time', 'Amount'], axis=1, inplace=True)
    return df_copy

X_train,X_test, y_train, y_test = get_train_test_dataset(df)
get_model_train_eval(lr, X_train, X_test, y_train, y_test)
print('\n')
get_model_train_eval(lgbm, X_train, X_test, y_train, y_test)

print("각 피처별 상관도 분석")
plt.figure(figsize=(9, 9))
corr = df.corr()
sns.heatmap(corr, cmap='RdBu')
plt.show()

def get_outlier(df, column, weight=1.5):
    fraud = df[df['Class'] == 1][column]
    quantile_25 = np.percentile(fraud.values, 25)
    quantile_75 = np.percentile(fraud.values, 75)
    iqr = quantile_75 - quantile_25
    iqr_weight = iqr * weight
    lowest_val =  quantile_25 - iqr_weight
    highest_val = quantile_75 + iqr_weight
    outlier_index = fraud[(fraud < lowest_val) | (fraud > highest_val)].index
    return outlier_index

outlier_index = get_outlier(df, 'V14', 1.5)
print("outlier_index : ", outlier_index)

print('\n')

def get_preprocessed_df(df):
    df_copy = df.copy()
    amount_n = np.log1p(df_copy["Amount"])
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    df_copy.drop(['Time', 'Amount'], axis=1, inplace=True)
    outlier_index = get_outlier(df_copy, 'V14', 1.5)
    df_copy.drop(outlier_index, axis=0, inplace=True)
    return df_copy

X_train, X_test, y_train, y_test = get_train_test_dataset(df)
get_model_train_eval(lr, X_train, X_test, y_train, y_test)
print('\n')
get_model_train_eval(lgbm, X_train, X_test, y_train, y_test)

print('\n')

print("SMOTE 데이터셋 \n")
smote = SMOTE()
X_train_over, y_train_over = smote.fit_sample(X_train, y_train)
print("X_train.shape, y_train.shape : ", X_train.shape, y_train.shape)
print("X_train_over.shape, y_train_over.shape : ", X_train_over.shape, y_train_over.shape)
print("y_train_over_value_counts : \n", pd.Series(y_train_over).value_counts())

print("\n")

print("Logistic Regression \n")
lr = LogisticRegression()
get_model_train_eval(lr, X_train_over, X_test, y_train_over, y_test)

def precision_recall_curve_plot(y_test, pred_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_prob)
    plt.figure(figsize=(8, 6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary], label='recall')
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.legend()
    plt.grid()
    plt.show()

precision_recall_curve_plot(y_test, lr.predict_proba(X_test)[:, 1])

print("\n")

print("LightGBM \n")
lgbm = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=-1, boost_from_average=False)
get_model_train_eval(lgbm, X_train_over, X_test, y_train_over, y_test)