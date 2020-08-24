import pandas as pd
import gzip
import json

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

filepath = r'C:\Users\nrapa\Dropbox\text_explanations\Data\hidden_factors\small_datasets\reviews_Books_5.json.gz'
df = getDF(filepath)