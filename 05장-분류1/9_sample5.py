import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns

# 사용자 행동 인식
df = pd.read_csv('./data/UCI_HAR_Dataset/UCI HAR Dataset/features.txt',
                 sep='\s+',     # \s : 공백 구분, + : 1개이상일떄
                 header=None,
                 names=['column_index', 'column_name'])

feature_name = df.iloc[:, 1].values.tolist()
print("feature_name : ", feature_name[:10])

print("\n")

# 중복된 피처 확인
feature_dup = df.groupby('column_name').count()
print("feature_dup : ", feature_dup[feature_dup['column_index'] > 1].count())
print("feature_dup 중복 : ", feature_dup[feature_dup['column_index'] > 1].head())

print("\n")

# 공백 및 전처리 함수 구현
def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(), columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how='outer')
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x: x[0] + '_' + str(x[1]) if x[1] > 0 else x[[0]], axis=1)
    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)
    return new_feature_name_df

def get_human_dataset():
    feature_name_df = pd.read_csv('./data/UCI_HAR_Dataset/UCI HAR Dataset/features.txt',
                 sep='\s+',     # \s : 공백 구분, + : 1개이상일떄
                 header=None,
                 names=['column_index', 'column_name'])
    new_feature_name_df = get_new_feature_name_df(feature_name_df)
    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()
    X_train = pd.read_csv('./data/UCI_HAR_Dataset/UCI HAR Dataset/train/X_train.txt', sep='\s+', names=feature_name)
    X_test = pd.read_csv('./data/UCI_HAR_Dataset/UCI HAR Dataset/test/X_test.txt', sep='\s+', names=feature_name)
    y_train = pd.read_csv('./data/UCI_HAR_Dataset/UCI HAR Dataset/train/y_train.txt', sep='\s+', header=None, names=['action'])
    y_test = pd.read_csv('./data/UCI_HAR_Dataset/UCI HAR Dataset/test/y_test.txt', sep='\s+', header=None, names=['action'])
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_human_dataset()

print("X_train_info : ", X_train.info())

print("\n")

print("y_train['action'] : \n", y_train['action'].value_counts())

print("\n")

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("accuracy : ", accuracy)

print("하이퍼 파라미터 값 : ", clf.get_params())

print("\n")

params = {'max_depth' : [6, 8, 10, 12, 16, 20, 24]}
grid_cv = GridSearchCV(clf, param_grid=params, scoring='accuracy', cv=5)
grid_cv.fit(X_train, y_train)
print("best_score : ", grid_cv.best_score_)
print("best_params : ", grid_cv.best_params_)

print("\n")

cv_results_df = pd.DataFrame(grid_cv.cv_results_)
print("cv_results_df : ", cv_results_df[['param_max_depth', 'mean_test_score']])

print("\n")

#최적의 하이퍼 파라미터 값 찾기
max_depths = [6, 8, 10, 12, 16, 20, 24]
for depth in max_depths:
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print("depth, accuracy :  ", depth, accuracy)

print("\n")

# 최적의 하이퍼 파라미터 값 찾기
params = {'max_depth' : [6, 8, 10, 12, 16, 20, 24], 'min_samples_split' : [16, 24]}
grid_cv = GridSearchCV(clf, param_grid=params, scoring='accuracy', cv=5)
grid_cv.fit(X_train, y_train)
print("best_score : ", grid_cv.best_score_)
print("best_params : ", grid_cv.best_params_)

best_clf = grid_cv.best_estimator_         # grid_cv.best_estimator_ 최적의 파라미터로 모델 생성
pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print("accuracy : ", accuracy)

print("\n")

importances_values = best_clf.feature_importances_
importances = pd.Series(importances_values, index=X_train.columns)
top20 = importances.sort_values(ascending=False)[:20]
sns.barplot(x=top20, y=top20.index)

plt.show()