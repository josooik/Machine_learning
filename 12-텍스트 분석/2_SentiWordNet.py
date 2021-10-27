from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import pandas as pd
import nltk

print("SentiWordNet \n")

synsets = list(swn.senti_synsets('slow'))
print("type(synsets) :", type(synsets))
print("len(synsets) :", len(synsets))
print("synsets :", synsets)

print("\n")

father = swn.senti_synset('father.n.01')
print("father.pos_score() : %f, father.neg_score() : %f, father.obj_score() : %f" % (father.pos_score(), father.neg_score(), father.obj_score()))

fabulous = swn.senti_synset('fabulous.a.01')
print("fabulous.pos_score() : %f, fabulous.neg_score() : %f, fabulous.obj_score() : %f" % (fabulous.pos_score(), fabulous.neg_score(), fabulous.obj_score()))

print("\n")

print("SentiWordNet – 영화 감상평 감성 분석 \n")
# 1.문서를 문장으로 분해
# 2.문장을 단어로 토큰화 및 품사 태깅
# 3.synset과 senti_synset 생성
# 4.긍정/부정 감성 결정

def penn_to_wn(tag):
    if tag.startswith('J'): return wn.ADJ
    elif tag.startswith('N'): return wn.NOUN
    elif tag.startswith('R'): return wn.ADV
    elif tag.startswith('V'): return wn.VERB

def swn_polarity(text):
    sentiment = 0.0
    tokens_count = 0
    lemmatizer = WordNetLemmatizer()
    sentences = sent_tokenize(text)
    for sentence in sentences:
        tagged_sentence = pos_tag(word_tokenize(sentences))
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)

            if wn_tag not in (wn.NOUN, wn.ADj, wn.ADV):
                continue

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)

            if not lemma:
                continue

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue

            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            sentiment += (swn_synset.pos_score() - swn_synset.neg_score())
            tokens_count += 1

    if not tokens_count:
        return 0

    if sentiment >= 0:
        return 1

    return 0

df['preds'] = df['review'].apply(lambda x: swn_polarity(x))
y_target = df['sentiment'].values
preds = df['preds'].values

print("confusion_matrix(y_target, preds) :\n", confusion_matrix(y_target, preds))
print("accuracy_score(y_target, preds) :", accuracy_score(y_target, preds))
print("precision_score(y_target, preds) :", precision_score(y_target, preds))
print("recall_score(y_target, preds) :", recall_score(y_target, preds))