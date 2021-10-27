import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

print("COO (Coordinate) 형식 \n")

dense = np.array([ [3, 0, 1], [0, 2, 0]])

data = np.array([3, 1, 2])
rows = np.array([0, 0, 1])
cols = np.array([0, 2, 1])

sparse_coo = sparse.coo_matrix((data, (rows, cols)))
print("sparse_coo : \n", sparse_coo)

print("\n")

print("sparse_coo.toarray() : \n", sparse_coo.toarray())

print("\n")

print("CSR (Compressed Sparse Row) 형식 \n")

data2 = np.array([1, 5, 1, 4, 3, 2, 5, 6, 3, 2, 7, 8, 1])
rows = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 4, 4, 5])
cols = np.array([2, 5, 0, 1, 3, 4, 5, 1, 3, 0, 3, 5, 0])

sparse_coo = sparse.coo_matrix((data2, (rows, cols)))
rows_index = np.array([0, 2, 7, 9, 10, 12, 13])
sparse_csr = sparse.csr_matrix((data2, cols, rows_index))

print("sparse_coo.toarray() : \n", sparse_coo.toarray())

print("\n")

print("sparse_csr.toarray() : \n", sparse_csr.toarray())

print("\n")

dense = np.array([[0, 0, 1, 0, 0, 5],
                  [1, 4, 0, 3, 2, 5],
                  [0, 6, 0, 3, 0, 0],
                  [2, 0, 0, 0, 0, 0],
                  [0, 0, 0, 7, 0, 8],
                  [1, 0, 0, 0, 0, 0]])

print("sparse.coo_matrix(dense) : \n", sparse.coo_matrix(dense))

print("\n")

dense = np.array([[0, 0, 1, 0, 0, 5],
                  [1, 4, 0, 3, 2, 5],
                  [0, 6, 0, 3, 0, 0],
                  [2, 0, 0, 0, 0, 0],
                  [0, 0, 0, 7, 0, 8],
                  [1, 0, 0, 0, 0, 0]])

print("sparse.csr_matrix(dense) : \n", sparse.csr_matrix(dense))

print("\n")

print("뉴스그룹 분류 – 텍스트 정규화 \n")

news = fetch_20newsgroups(subset='all')

print("news.keys() : ", news.keys())

print("\n")

print(pd.Series(news.target).value_counts().sort_index())

print("\n")

print(news.target_names)

print("\n")

print(news.data[0])

news_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
X_train = news_train.data
y_train = news_train.target

news_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
X_test = news_test.data
y_test = news_test.target

print("news_train_data : %d, news_test_data : %d" %(len(news_train.data), len(news_test.data)))

print("\n")

print("뉴스그룹 분류 – 피처 벡터화 및 학습 \n")

cnt_vect = CountVectorizer()
cnt_vect.fit(X_train)
X_train_cnt_vect = cnt_vect.transform(X_train)
X_test_cnt_vect = cnt_vect.transform(X_test)

print(X_train_cnt_vect.shape)

print("\n")

print("뉴스그룹 분류 – 평가 \n")

lr = LogisticRegression()
lr.fit(X_train_cnt_vect, y_train)
pred = lr.predict(X_test_cnt_vect)
print("LogisticRegression accuracy_score : ", accuracy_score(y_test, pred))

print("\n")

tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(X_train)
X_train_tfidf_vect = tfidf_vect.transform(X_train)
X_test_tfidf_vect = tfidf_vect.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_tfidf_vect, y_train)
pred = lr.predict(X_test_tfidf_vect)
print("TfidfVectorizer LogisticRegression accuracy_score : ", accuracy_score(y_test, pred))

print("\n")

tfidf_vect = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=300)
tfidf_vect.fit(X_train)
X_train_tfidf_vect = tfidf_vect.transform(X_train)
X_test_tfidf_vect = tfidf_vect.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_tfidf_vect, y_train)
pred = lr.predict(X_test_tfidf_vect)
print("TfidfVectorizer stop_words accuracy_score : ", accuracy_score(y_test, pred))

print("\n")

params = {'C' : [0.01, 0.1, 1, 5, 10]}
grid_cv = GridSearchCV(lr, param_grid=params, cv=3, scoring='accuracy')
grid_cv.fit(X_train_tfidf_vect, y_train)
print("grid_cv.best_params_ : ", grid_cv.best_params_)

pred = grid_cv.predict(X_test_tfidf_vect)
print("\n")
print("GridSearchCV accuracy_score : ", accuracy_score(y_test, pred))