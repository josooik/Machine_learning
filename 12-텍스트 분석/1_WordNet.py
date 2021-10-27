import nltk
import pandas as pd
from nltk.corpus import wordnet as wn

print("WordNet \n")

#nltk.download('all')
print("\n")

print("present 단어에 대한 Synset 추출 \n")
synsets = wn.synsets('present')
print("type(synsets) :", type(synsets))
print("len(synsets) :", len(synsets))
print("synsets :", synsets)

print("\n")

for synsets in synsets:
    print(synsets.name())           # POS태그 이름
    print(synsets.lexname())        # 어휘 이름
    print(synsets.definition())     # 단어의 뜻
    print(synsets.lemma_names())     # 부명제
    print('-------')

print("\n")

print("어휘 간의 유사도 path_similarity() \n")

tree = wn.synset('tree.n.01')
lion = wn.synset('lion.n.01')
tiger = wn.synset('tiger.n.02')
cat = wn.synset('cat.n.01')
dog = wn.synset('dog.n.01')

entities = [tree, lion, cat, dog]
similarities = []
entity_names = [entity.name().split('.')[0] for entity in entities]

for entity in entities:
    similarity = [entity.path_similarity(ent) for ent in entities]
    similarities.append(similarity)

df = pd.DataFrame(similarities, columns=entity_names, index=entity_names)
print(df)