import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

titanic_df = pd.read_csv('./data/titanic/train.csv')
print(titanic_df.head(3))
print("\n")

print(titanic_df.info())
print("\n")
print(titanic_df.isnull().sum())
print("\n")

titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
titanic_df['Cabin'].fillna('N', inplace=True)
titanic_df['Embarked'].fillna('N', inplace=True)
print(titanic_df.isnull().sum())
print("\n")

print(titanic_df['Sex'].value_counts())
print("\n")
print(titanic_df['Cabin'].value_counts())
print("\n")
print(titanic_df['Embarked'].value_counts())
print("\n")

titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
print(titanic_df['Cabin'].value_counts())
print("\n")

print(titanic_df.groupby(['Sex', 'Survived'])['Survived'].count())
print("\n")

sns.barplot(x='Sex', y='Survived', data=titanic_df)

sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)

def get_category(age):
    cat = ''
    if age <= -1: cat = 'Unknown'
    elif age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else: cat = 'Elderly'
    return cat

plt.figure(figsize=(10, 6))
group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Elderly']

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda  x: get_category(x))
sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titanic_df, order=group_names)
titanic_df.drop('Age_cat', axis=1, inplace=True)

def encode_features(df):
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

titanic_df = encode_features(titanic_df)
print(titanic_df.head())
