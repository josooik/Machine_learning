import pandas as pd
import glob, os
from nltk.stem import WordNetLemmatizer
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

print("문서 군집화 \n")

all_files = glob.glob(os.path.join('./data/opinion/', '*.data'))
filenames = []
texts = []

for f in all_files:
    df = pd.read_table(f, index_col=None, header=0, encoding='latin1')
    fname = f.split('\\')[-1]
    fname = fname.split('.')[0]
    filenames.append(fname)
    texts.append(df.to_string())

df = pd.DataFrame({'filename': filenames, 'opinion_text': texts})
print(df.head())

print("\n")

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
print("string.punctuation :", string.punctuation)
print("remove_punct_dict :", remove_punct_dict)

lemmar = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmar.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

tfidf_vect = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', ngram_range=(1, 2), min_df=0.05, max_df=0.85)

feature_vect = tfidf_vect.fit_transform(df['opinion_text'])

kmeans = KMeans(n_clusters=5, max_iter=10000)
kmeans.fit(feature_vect)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

df['cluster'] = labels
print(df.head())

print("\n")

print(df[df['cluster'] == 0].sort_values(by='filename'))

print("\n")

print(df[df['cluster'] == 1].sort_values(by='filename'))

print("\n")

kmeans = KMeans(n_clusters=3, max_iter=10000)
kmeans.fit(feature_vect)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

df['cluster'] = labels
print(df.sort_values(by='cluster'))

print("\n")

centers = kmeans.cluster_centers_
print("centers.shape :", centers.shape)
print("\n")
print("center :\n", centers)

print("\n")

print("군집별 핵심 단어 추출 \n")

def get_cluster_details(model, data, feature_names, n_clusters, top_n_features=10):
    details = {}
    ordered_index = model.cluster_centers_.argsort()[:, ::-1]
    for n in range(n_clusters):
        details[n] = {}
        details[n]['cluster'] = n
        top_feature_indices = ordered_index[n, :top_n_features]
        top_features = [feature_names[index] for index in top_feature_indices]
        top_feature_values = model.cluster_centers_[n, top_feature_indices].tolist()

        details[n]['top_features'] = top_features
        details[n]['top_fefatures_value'] = top_feature_values
        filenames = data[data['cluster'] == n]['filename']
        filenames = filenames.values.tolist()
        details[n]['filenames'] = filenames

    return details

def print_cluster_details(details):
    for n, detail in details.items():
        print(n)
        print(detail['top_features'])
        print(detail['filenames'][:7])
        print('-----')

feature_names = tfidf_vect.get_feature_names()
details = get_cluster_details(kmeans, df, feature_names, 3, 10)
print_cluster_details(details)
