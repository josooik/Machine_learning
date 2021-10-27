from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

print("LAD \n")

cats = ['rec.motorcycles', 'rec.sport.baseball', 'comp.graphics', 'comp.windows.x', 'talk.politics.mideast', 'soc.religion.christian', 'sci.electronics', 'sci.med']
df = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), categories=cats)
count_vect = CountVectorizer(max_df=0.95, max_features=1000, min_df=2, stop_words='english', ngram_range=(1, 2))
feat_vect = count_vect.fit_transform(df.data)
print("feat_vect.shape :", feat_vect.shape)
print("\n")

lda = LatentDirichletAllocation(n_components=8)
lda.fit(feat_vect)
print("lda.components_.shape :", lda.components_.shape)
print("\n")
print("lda.components_ :\n", lda.components_)

print("\n")

def display_topics(model, feature_names, no_top_words):
    for topic_index, topic in enumerate(model.components_):
        print("topic_index :", topic_index)
        topic_word_indices = topic.argsort()[::-1]
        top_indices = topic_word_indices[:no_top_words]
        feature_concat = ' '.join([feature_names[i] for i in top_indices])
        print("feature_concat :", feature_concat)

feature_names = count_vect.get_feature_names()
display_topics(lda, feature_names, 15)