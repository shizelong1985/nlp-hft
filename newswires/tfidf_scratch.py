import json
import numpy as np
from collections import Counter
from string import punctuation
import nltk
# from porter2stemmer import Porter2Stemmer
import re
from nltk.stem import WordNetLemmatizer

from nltk import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

with open(r'C:\Users\nrapa\git\nlp-hft\data\DowJones_Newswires\sample\sample_tagged_single.txt') as json_file:
    data = json.load(json_file)

# Define stopwords
stopwords = nltk.corpus.stopwords.words("english") + list(punctuation)


def tokenize(text):
    # tokenize
    words = word_tokenize(text)

    # remove all characters of length 1
    words = [w for w in words if len(w) > 1]

    # remove URLs
    words = [re.sub(r"http\S+", "", w) for w in words]
    words = [re.sub(r"www.+", "", w) for w in words]

    # remove punctuation and special symbols
    for i in punctuation:
        words = [''.join([char for char in word if char != i]) for word in words]

    # convert to lower case
    words = [w.lower() for w in words]

    # remove common stop words and numbers
    tokens = [w for w in words if w not in stopwords and not w.isdigit()]

    # lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = []
    for item in tokens:
        lemmas.append(wordnet_lemmatizer.lemmatize(item))

    # stemming
    # stemmer = Porter2Stemmer()
    # stems = []
    # for item in lemmas:
    #     stems.append(stemmer.stem(item))

    return lemmas

# Define corpus
date = '2020-03-26'
documents = [el['text'] for el in data[date]]


documents = [tokenize(text) for text in documents]

# Document Frequency
dfs = {}
for i in range(len(documents)):
    for word in documents[i]:
        try:
            dfs[word].add(i)
        except:
            dfs[word] = {i}

for i in dfs:
    dfs[i] = len(dfs[i])

# Tf-Idf
tf_idf = {}
for i in range(len(documents)):
    counter = Counter(documents[i])
    for token in np.unique(documents[i]):
        tf = counter[token]/len(documents[i])
        df = dfs[token]
        # idf = np.log(len(documents)/(df+1))
        idf = np.log(len(documents)/(df))
        tf_idf[i, token] = tf*idf

X = pd.DataFrame(list(tf_idf.items())).sort_values(by=[1], ascending=False)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

word_cloud_dict = {}
for key, val in tf_idf.items():
    if key[1] not in word_cloud_dict or val > word_cloud_dict[key[1]]:
        word_cloud_dict[key[1]] = val

wc = WordCloud(
    background_color="white",
    max_words=2000,
    width=1024,
    height=720,
)

wordcloud = wc.generate_from_frequencies(word_cloud_dict)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
