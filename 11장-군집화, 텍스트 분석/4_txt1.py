import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer


print("토큰화 \n")

nltk.download('punkt')

text_sample = 'The Matrix is everywhere its all around us, here event in this room. \
You can see it out your window or on your television. \
You feel it  when you go to work, or go to church or pay your taxes.'

sentences = sent_tokenize(text=text_sample)

print("\n")

print(type(sentences), len(sentences))

print("\n")

print(sentences)

sentence = 'The Matrix is everywhere its all around us, here even in this room.'
words = word_tokenize(sentence)

print("\n")

print(type(words), len(words))

print("\n")

print(words)

def tokenize_text(text):
    sentences = sent_tokenize(text)
    return [word_tokenize(sentence) for sentence in sentences]

word_tokens = tokenize_text(text_sample)

print("\n")

print(type(word_tokens), len(word_tokens))

print("\n")

print(word_tokens)

print("\n")

print("스톱 워드 \n")

nltk.download('stopwords')

print("\n")

print(len(nltk.corpus.stopwords.words('english')))

print("\n")

print(nltk.corpus.stopwords.words('english')[:20])

print("\n")

print("스톱 워드 제거 \n")

stopwords = nltk.corpus.stopwords.words('english')
all_tokens = []
for sentence in word_tokens:
    filtered_words = []
    for word in sentence:
        word = word.lower()
        if word not in stopwords:
            filtered_words.append(word)

    all_tokens.append(filtered_words)

print(all_tokens)

print("\n")

print("Stemming/Lemmatization 단어의 원형 찾기 \n")

stemmer = LancasterStemmer()
print(stemmer.stem('working'), stemmer.stem('works'), stemmer.stem('worked'))
print(stemmer.stem('amusing'), stemmer.stem('amuses'), stemmer.stem('amused'))
print(stemmer.stem('happier'), stemmer.stem('happiest'))
print(stemmer.stem('fancier'), stemmer.stem('fanciest'))

print("\n")

nltk.download('wordnet')

lemma = WordNetLemmatizer()
print("\n")
print(lemma.lemmatize('amusing', 'v'), lemma.lemmatize('amuses', 'v'), lemma.lemmatize('amused', 'v'))
print(lemma.lemmatize('happier', 'a'), lemma.lemmatize('happiest', 'a'))
print(lemma.lemmatize('fancier', 'a'), lemma.lemmatize('fanciest', 'a'))