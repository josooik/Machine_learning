import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import nltk

print("IMDB 영화평 – 지도학습 기반 감성 분석 \n")

df = pd.read_csv('./data/imdb/imdb.tsv', sep='\t', quoting=3)
print(df.head(3))

print("\n")

print(df['review'][0])

print("\n")

df['review'] = df['review'].str.replace('<br />', ' ')                      # <br /> 태그를 공백으로 변경
df['review'] = df['review'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))    # 숫자/특수문자를 공백으로 변경

class_df = df['sentiment']
feature_df = df.drop(['id', 'sentiment'], axis=1, inplace=False)
X_train, X_test, y_train, y_test = train_test_split(feature_df, class_df, test_size=0.3)
print("X_train.shape : %s, X_test.shape : %s" %(X_train.shape, X_test.shape))

print("\n")

print("CountVectorizer \n")

pipeline = Pipeline([
    ('cnt_vect', CountVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('lr', LogisticRegression(C=10))
])

pipeline.fit(X_train['review'], y_train)
pred = pipeline.predict(X_test['review'])
pred_prob = pipeline.predict_proba(X_test['review'])[:,1]
print("accuracy_score : %f, roc_auc_score : %f" %(accuracy_score(y_test, pred), roc_auc_score(y_test, pred_prob)))

print("\n")

print("TfidfVectorizer \n")

pipeline = Pipeline([
    ('tfidf_vect', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('lr', LogisticRegression(C=10))
])
pipeline.fit(X_train['review'], y_train)
pred = pipeline.predict(X_test['review'])
pred_prob = pipeline.predict_proba(X_test['review'])[:,1]
print("accuracy_score : %f, roc_auc_score : %f" %(accuracy_score(y_test, pred), roc_auc_score(y_test, pred_prob)))