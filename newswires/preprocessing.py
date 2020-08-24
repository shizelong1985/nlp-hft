import json

with open(r'C:\Users\nrapa\git\nlp-hft\data\DowJones_Newswires\sample\sample.txt') as json_file:
    data = json.load(json_file)


def count_documents(data):
    n_documents = [len(data[date]) for date in data]
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
n_doc = count_documents(data)
print(f'Total Number of Documents: {sum(n_doc.values())}')

n_doc = count_documents(data_tagged)
print(f'Total Number of Documents with at least one firm tagged: {sum(n_doc.values())}')

n_doc = count_documents(data_tagged_single)
print(f'Total Number of Documents with a single firm tagged: {sum(n_doc.values())}')

# Save
with open(r'C:\Users\nrapa\git\nlp-hft\data\DowJones_Newswires\sample\sample_tagged.txt', 'w') as outfile:
    json.dump(data_tagged, outfile)

with open(r'C:\Users\nrapa\git\nlp-hft\data\DowJones_Newswires\sample\sample_tagged_single.txt', 'w') as outfile:
    json.dump(data_tagged_single, outfile)

