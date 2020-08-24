import json
from string import punctuation
import nltk
from nltk import word_tokenize
import re
from porter2stemmer import Porter2Stemmer
import contractions
from nltk.stem import WordNetLemmatizer

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

with open(r'../data/DowJones_Newswires/sample/sample_tagged_single.txt') as json_file:
    data = json.load(json_file)

# Define stopwords
nltk.download('punkt')
nltk.download('stopwords')  # (Item 70 from http://www.nltk.org/nltk_data/)
nltk.download('words')
nltk.download('wordnet')

symbols = list(punctuation)
stopwords = nltk.corpus.stopwords.words("english")

more_stopwords = [
    'end', 'et', 'gmt',
    'benzinga', 'ratings', 'action', 'actions', 'follow',
    'dow', 'jones', 'trading', 'newswires', 'source',
    'follow', 'company', 'service', 'rating', 'info', 'insider',
    'business', 'inc', 'sell', 'buy', 'unch',
]
stopwords = stopwords + more_stopwords


def tokenize(text):
    # tokenize
    words = word_tokenize(text)

    # remove URLs
    words = [w for w in words if not w[-4:] == '.com']
    words = [re.sub(r"http\S+", "", w) for w in words]
    words = [re.sub(r"www.+", "", w) for w in words]

    # remove punctuation and special symbols
    for i in punctuation:
        words = [''.join([char for char in word if char != i]) for word in words]

    # remove words containing numbers
    words = [word for word in words if word.isalpha()]

    # convert to lower case
    words = [w.lower() for w in words]

    # fix contractions
    words = [contractions.fix(w) for w in words]

    # remove common stop words
    tokens = [w for w in words if w not in stopwords]

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

    stems = lemmas

    # keep stems with length greater than 1
    stems = [s for s in stems if len(s) > 1]

    return stems


# Define corpus
date = '2020-03-26'
corpus = [el['text'] for el in data[date]]

vectorizer = TfidfVectorizer(
    sublinear_tf=False,
    # min_df=5,
    max_df=0.5,
    norm='l2',
    ngram_range=(1, 2),
    smooth_idf=False,
    tokenizer=tokenize,
    use_idf=True,
)

tdm = vectorizer.fit_transform(corpus)
df = pd.DataFrame(data=tdm.toarray(), columns=vectorizer.get_feature_names())

# Plot Word Cloud
wc = WordCloud(
    background_color="white",
    max_words=2000,
    width=1024,
    height=720,
)

X = df.T.mean(axis=1)
wc.generate_from_frequencies(X)
plt.imshow(wc)
