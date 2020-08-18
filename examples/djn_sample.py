import json

import nltk
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

with open(r'C:\Users\nrapa\git\nlp-hft\data\DowJones_Newswires\sample\sample.txt') as json_file:
    data = json.load(json_file)

corpus = [el['text'] for el in data['2020-03-23']]

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words("english")
more_stopwords = [
    # 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
    # 'november', 'december',
    # 'end', 'et', 'gmt',
    'benzinga', 'ratings', 'action', 'actions', 'follow',
    # '2020', '23', 'com',
    # 'dow', 'jones', 'trading', 'newswires', 'www', 'source', 'https'
]
stopwords = stopwords + more_stopwords

vectorizer = TfidfVectorizer(
    sublinear_tf=False,
    min_df=5,
    norm='l2',
    encoding='latin-1',
    ngram_range=(1, 2),
    stop_words=stopwords,
    smooth_idf=False,
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

# plt.show()
#
# w = WordCloud(width=800,height=600,mode='RGBA',background_color='white',max_words=200).fit_words(freqs)
#
#
# # print(vectorizer.get_feature_names())
# #
# # print(tdm.shape)
#
# # documents = [doc.replace('\n', ' ').split() for doc in documents]
# #
# # documents = [[word for word in sublist if (len(word)>1)]
# #              for sublist in documents]
# #
# # documents = [[word.lower() for word in sublist]
# #              for sublist in documents]
# #
# # symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\'"
# # for i in symbols:
# #     documents = [[''.join([char for char in word if char != i])
# #                   for word in sublist]
# #                   for sublist in documents]
# #
# # dfs = {}
# # for i in range(len(documents)):
# #     for word in documents[i]:
# #         try:
# #             dfs[word].add(i)
# #         except:
# #             dfs[word] = {i}
# #
# # for i in dfs:
# #     dfs[i] = len(dfs[i])
# #
# # tf_idf = {}
# # for i in range(len(documents)):
# #     counter = Counter(documents[i])
# #     for token in np.unique(documents[i]):
# #         tf = counter[token]/len(documents[i])
# #         df = dfs[token]
# #         idf = np.log(len(documents)/(df+1))
# #         tf_idf[i, token] = tf*idf
# #
# # # Visualization
# #
# # word_cloud_dict = {}
# # for key, val in tf_idf.items():
# #     if key[1] not in word_cloud_dict or val > word_cloud_dict[key[1]]:
# #         word_cloud_dict[key[1]] = val
# #
# # wordcloud = WordCloud().generate_from_frequencies(word_cloud_dict)
# #
# # plt.imshow(wordcloud, interpolation='bilinear')
# # plt.axis("off")
# # plt.show()
