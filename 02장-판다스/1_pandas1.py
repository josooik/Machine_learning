import pandas as pd
import numpy as np

titanic_df = pd.read_csv('./data/titanic/train.csv')
print(titanic_df.head(3))
print(type(titanic_df))
print(titanic_df)
print(titanic_df.shape)
print(titanic_df.info())
print(titanic_df.describe())

pclass = titanic_df['Pclass']
print(pclass)

print(pclass.value_counts())

col_name1 = ['coll']
list1 = [1, 2, 3]
df_list1 = pd.DataFrame(list1, columns=col_name1)
print(df_list1)

col_name1 = ['coll']
array1 = np.array([1, 2, 3])
print(array1.shape)
df_array1 = pd.DataFrame(array1, columns=col_name1)
print(df_list1)

col_name2 = ['col1', 'col2', 'col3']
list2 = [[1, 2, 3], [11, 12, 13]]
df_list2 = pd.DataFrame(list2, columns=col_name2)
print(df_list2)

col_name2 = ['col1', 'col2', 'col3']
array2 = np.array([[1, 2, 3], [11, 12, 13]])
print(array2.shape)
df_array2 = pd.DataFrame(array2, columns=col_name2)
print(df_array2)

dict = {'col1' : [1, 11], 'col2' : [2, 22], 'col3' : [3, 33]}
df_dict = pd.DataFrame(dict)
print(df_dict)

array3 = df_dict.values
print(type(array3), array3.shape)
print(array3)

list3 = df_dict.values.tolist()
print(type(list3))
print(list3)

dict3 = df_dict.to_dict('list')
print(type(dict3))
print(dict3)

titanic_df['Age_0'] = 0
print(titanic_df.head(3))

titanic_df['Age_by_10'] = titanic_df['Age'] * 10
titanic_df['Family_No'] = titanic_df['SibSp'] + 100
print(titanic_df.head(3))

titanic_df['Age_by_10'] = titanic_df['Age_by_10'] + 100
print(titanic_df.head(3))

drop_df = titanic_df.drop('Age_0', axis=1)
print(drop_df.head(3))

print(titanic_df.head(3))

drop_df = titanic_df.drop(['Age_0', 'Age_by_10', 'Family_No'], axis=1, inplace=True)
print(drop_df)
print(titanic_df.head(3))

titanic_df.drop([0, 1, 2], axis=0, inplace=True)
print(titanic_df.head(3))

titanic_df = pd.read_csv('./data/titanic/train.csv')
indices = titanic_df.index
print(indices)

print(indices.values)

print(type(indices.values))
print(indices.values.shape)
print(indices[:5].values)
print(indices.values[:5])
print(indices[6])

series_fair = titanic_df['Fare']
print(series_fair)

print(series_fair.max())
print(series_fair.min())
print(series_fair.sum())
print(sum(series_fair))
print((series_fair + 3).head(3))

reset_df = titanic_df.reset_index(inplace=False)
print(reset_df.head(3))

value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
print(type(value_counts))
new_value_counts = value_counts.reset_index(inplace=False)
print(new_value_counts)
print(type(new_value_counts))

reset_df = titanic_df.reset_index(inplace=False, drop=True)
print(reset_df.head(3))

value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
print(type(value_counts))
new_value_counts = value_counts.reset_index(inplace=False, drop=True)
print(new_value_counts)
print(type(new_value_counts))

print(titanic_df['Pclass'].head(3))

print(titanic_df[['Survived', 'Pclass']].head(3))

print(titanic_df[titanic_df['Pclass'] == 3].head(3))

print(titanic_df[0:2])

data = {'Name' : ['Chulmin', "Eunkyung", 'Jinwoong', 'Soobecom'],
        'Year' : [2001, 2016, 2015, 2015],
        'Gender' : ['Male', 'Female', 'Male', 'Male']}
data_df = pd.DataFrame(data, index = ['one', 'two', 'three', 'four'])
print("\n", data_df)

print("\n", data_df.iloc[0, 0])

print("\n", data_df.iloc[[0, 2], 0])

print("\n", data_df.iloc[[0, 2], [0, 2]])

print("\n", data_df.iloc[1:3, 0])

print("\n", data_df.iloc[2:, 1:3])

print("\n", data_df.loc['one', 'Name'])

reset_df = data_df.reset_index()
reset_df.index = reset_df.index + 1
print("\n", reset_df)

print("\n", reset_df.loc[1, 'Name'])

print("\n", data_df.loc[['one', 'three'], 'Name'])

print("\n", data_df.loc[['one', 'three'], ['Name', 'Gender']])

print("\n", data_df.loc['one' : 'three', 'Name'])

print("\n", data_df.loc['one' : 'three', 'Name' : 'Year'])

titanic_boolean = titanic_df[titanic_df['Age'] > 60]
print("\n", type(titanic_boolean))
print("\n", titanic_boolean)

print("\n", titanic_df[titanic_df['Age'] > 60][['Name', 'Age']].head(3))

print("\n", titanic_df.loc[titanic_df['Age'] > 60, ['Name', 'Age']].head(3))

print("\n", titanic_df[(titanic_df['Age'] > 60) & (titanic_df['Pclass'] == 1) & (titanic_df['Sex'] == 'female')])

cond1 = titanic_df['Age'] > 60
cond2 = titanic_df['Pclass'] == 1
cond3 = titanic_df['Sex'] == 'female'
print("\n", titanic_df[cond1 & cond2 & cond3])

titanic_sorted = titanic_df.sort_values(by=['Name'])
print("\n", titanic_sorted.head(3))

titanic_sorted = titanic_df.sort_values(by=['Pclass', 'Name'], ascending=True, inplace=False)
print("\n", titanic_sorted.head(3))

print("\n", titanic_df.count())

print("\n", titanic_df[['Age', 'Fare']].mean())

titanic_groupby = titanic_df.groupby(by='Pclass')
print("\n", type(titanic_groupby))

print("\n", titanic_df.groupby('Pclass').count())

print("\n", titanic_df.groupby('Pclass')[['PassengerId', 'Survived']].count())

print("\n", titanic_df.groupby(['Pclass', 'Sex']).count())

print("\n", titanic_df.groupby(['Pclass', 'Sex'])[['PassengerId', 'Survived']].count())

print("\n", titanic_df.groupby('Pclass')['Age'].agg([max, min]))

print("\n", titanic_df.groupby('Pclass')[['Age', 'Fare']].agg([max, min]))

print("\n", titanic_df.groupby('Pclass').agg({'Age': 'max', 'SibSp': 'sum', 'Fare': 'mean'}))

print("\n", titanic_df.isna())

print("\n", titanic_df.isna().sum())

titanic_df['Cabin'] = titanic_df['Cabin'].fillna('C000')
print("\n", titanic_df.head(3))

titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df['Embarked'].fillna('S', inplace=True)
print("\n", titanic_df.isna().sum())

def get_square(a):
    return a**2
print("\n", get_square(3))

lambda_square = lambda x: x**2
print(lambda_square(3))

print("\n", list(map(lambda x: x**2, [1, 2, 3])))

titanic_df['Name_len'] = titanic_df['Name'].apply(lambda x: len(x))
print(titanic_df[['Name', 'Name_len']].head(3))

titanic_df['Child_Adult'] = titanic_df['Age'].apply(lambda x : 'Child' if x <= 15 else 'Adult')
print("\n", titanic_df[['Age', 'Child_Adult']].head(8))

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x: 'Child' if x <= 15 else ('Adult' if x <= 60 else 'Elderly'))
print("\n", titanic_df['Age_cat'].value_counts())

def get_category(age):
    cat = ''
    if age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else: cat = 'Elderly'
    return cat

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x: get_category(x))
print("\n", titanic_df[['Age', 'Age_cat']].head())