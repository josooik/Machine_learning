from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import time
import pandas as pd

# GBM (Gradient Boosting Machine)
print("GBM (Gradient Boosting Machine)")

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
start_time = time.time()

clf = GradientBoostingClassifier()
clf.fit(X_train, y_train.values.ravel())
pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("accuracy, time : ", accuracy, time.time() - start_time)

print("\n")

# 하이퍼 파라미터 튜닝
print("하이퍼 파라미터 튜닝")

params = {
    'n_estimators' : [100, 500],
    'learning_rate' : [0.05, 0.1]
}

grid_cv = GridSearchCV(clf, param_grid=params, cv=2)
grid_cv.fit(X_train, y_train.values.ravel())
print("best_params : ", grid_cv.best_params_)
print("best_score : ", grid_cv.best_score_)

pred = grid_cv.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print("accuracy : ", accuracy)