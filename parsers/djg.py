import json

from bs4 import BeautifulSoup
from tqdm import tqdm


def parse_document(doc):
    document = {}
    document['text'] = doc.body.text
    document['seq'] = doc.get('seq')
    document['publisher'] = doc.get('publisher')
    document['product'] = doc.get('product')

    # mdata = doc.find('djn-mdata')
    # date = mdata.get('display-date')
    # document['date'] = pd.to_datetime(date).tz_convert(None).tz_localize('GMT')
    # document['docdate'] = pd.to_datetime(doc.get('docdate'))

    try:
        isins = doc.find('djn-isin')
        document['isins'] = [el.get_text() for el in isins.find_all('c')]
    except:
        pass

    return document


dates = ['2020-03-23', '2020-03-24', '2020-03-25', '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30']

# TODO: Consider parallelization using joblib
data = {}
for date in tqdm(dates):
    # Initialize data storage
    data[date] = []

    # Open file
    filepath = fr'C:\Users\nrapa\git\nlp-hft\data\DowJones_Newswires\sample\DJG-US-20200323-20200330\{date}.nml'
    infile = open(filepath)
    contents = infile.read()

    # Parse file using BeautifulSoup
    soup = BeautifulSoup(contents)

    # Find all documents
    docs = soup.find_all('djnml')

    # Loop over all documents
    for doc in tqdm(docs):
        data[date].append(parse_document(doc))

with open(r'C:\Users\nrapa\git\nlp-hft\data\DowJones_Newswires\sample\sample.txt', 'w') as outfile:
    json.dump(data, outfile)
