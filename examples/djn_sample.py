import json
from string import punctuation
import nltk
from nltk import word_tokenize
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

with open(r'C:\Users\nrapa\git\nlp-hft\data\DowJones_Newswires\sample\sample.txt') as json_file:
    data = json.load(json_file)

def count_documents(data):
    n_documents = [len(data[date]) for date in data]
    print(f'Total Number of Documents: {sum(n_documents)}')
    n_documents = dict(zip(data.keys(), n_documents))
    return n_documents

# Remove articles with no stocks tagged
data_tagged = {}
data_tagged_single = {}
for date in data:
    data_tagged[date] = []
    data_tagged_single[date] = []
    for doc in data[date]:
        if doc['isins'] is not None:
            data_tagged[date].append(doc)
            if len(doc['isins']) == 1:
                doc['isin'] = doc['isins'][0]
                data_tagged_single[date].append(doc)

# Count documents
count_documents(data)
count_documents(data_tagged)
count_documents(data_tagged_single)

del data
del data_tagged

# Define stopwords
nltk.download('punkt')
nltk.download('stopwords') #(Item 70 from http://www.nltk.org/nltk_data/)
nltk.download('words')
nltk.download('wordnet')
stopwords = nltk.corpus.stopwords.words("english") + list(punctuation)

more_stopwords = [
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
    'november', 'december',
    'end', 'et', 'gmt',
    'benzinga', 'ratings', 'action', 'actions', 'follow',
    '2020', '23', 'com',
    'dow', 'jones', 'trading', 'newswires', 'www', 'source', 'https', 'http',
    'follow', 'company', 'service', 'statement', 'rating', '212-416-2800', 'info', 'insider',
    'business', 'source', 'inc.', 'from_', 'data_',  'date', 'data', 'form', 'dir', 'said', 'sell', 'buy',
]
stopwords = stopwords + more_stopwords

# from porter2stemmer import Porter2Stemmer
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def tokenize(text):
    # tokenize
    words = word_tokenize(text)
    # convert to lower case
    words = [w.lower() for w in words]
    # remove common stop words, delete numbers, punctuations, special symbols, and non-English words
    tokens = [w for w in words if w not in stopwords and not w.isdigit()]

    # lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    stems = []
    for item in tokens:
        stems.append(wordnet_lemmatizer.lemmatize(item))

    # stemming
    # stemmer = Porter2Stemmer()
    # stems = []
    # for item in tokens:
    #     stems.append(stemmer.stem(item))

    return stems

# Define corpus
date ='2020-03-26'
corpus = [el['text'] for el in data_tagged_single[date]]

# From scratch
# ###########################
# preprocessing

# corpus = [tokenize(text) for text in corpus]
#
#
# dfs = {}
# for i in range(len(corpus)):
#     for word in corpus[i]:
#         try:
#             dfs[word].add(i)
#         except:
#             dfs[word] = {i}
#############################################


vectorizer = TfidfVectorizer(
    sublinear_tf=False,
    # min_df=5,
    # max_df=0.5,
    norm='l2',
    # encoding='latin-1',
    ngram_range=(1,2),
    # stop_words=stopwords,
    smooth_idf=True,
    tokenizer=tokenize,
    use_idf=True,
)

tdm = vectorizer.fit_transform(corpus)

X = tdm.toarray()

df = pd.DataFrame(data=X, columns=vectorizer.get_feature_names())
tfidf_means = df.T.mean(axis=1)

wc = WordCloud(
    background_color="white",
    max_words=2000,
    width=1024,
    height=720,
)

wc.generate_from_frequencies(tfidf_means)
plt.imshow(wc)



