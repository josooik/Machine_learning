import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target
print("\n", iris_df)
print("\n", iris_df['label'].value_counts())
print("\n")

kfold = KFold(n_splits=3)
n_iter = 0
for train_index, test_index in kfold.split(iris_df):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print('\n iteration', n_iter)
    print("\n", label_train.value_counts())
    print("\n", label_test.value_counts())