# NOTES
# Remove relations
# Sort on start
# Multiple spans in one line
# random characters in ann file like "" (102027, line 410)
# Overlap max seq length for bert to preserve context, choose higher logit

import os
import csv
import nltk
import pandas as pd

nltk.download('punkt')

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR, '..', 'data', 'track2-training_data_2')
OUTPUT_DIR = os.path.join(CURR_DIR, '..', 'data', 'train')

def get_docs():
    for doc in os.listdir(DATA_DIR):
        patient = doc.split('.')[0]

        if doc.endswith('.txt'):
            if os.path.exists(os.path.join(DATA_DIR, patient + '.ann')):
                yield patient, \
                      os.path.join(DATA_DIR, patient + '.txt'), \
                      os.path.join(DATA_DIR, patient + '.ann'), \
                      os.path.join(OUTPUT_DIR, patient + '.csv')
            else:
                print('Annotation does not exist for txt file - ', doc)


def process(patient, txtpath, annpath, outpath):

    if os.path.exists(outpath):
        print('File already exists - ', outpath)
        return True

    def get_ann_list(annpath):
        print('Extracting data from ', annpath)
        df = pd.DataFrame()
        try:
            df = pd.read_csv(annpath, delimiter = '\t', names = ['id', 'ann', 'entity'])
        except Exception as e:
            print('Error while reading file - ', annpath, e)
            exit()

        df = df[df['id'].str.startswith("T")] # filter out R
        df[["tag", "start", "end"]] = df["ann"].str.split(pat=' ', n=2, expand=True) # split ann into tag-start-end
        df["end"] = df["end"].apply(lambda x: x.split(" ")[-1]) # take last pos of range(multiple spans when contd on next line)
        df['start'] = df['start'].astype(int) # string -> int
        df['end'] = df['end'].astype(int) # string -> int
        df.sort_values("start", inplace=True) # sort on start indices for pos match

        print('Extracted data from ', annpath)
        return df["tag"].tolist(), df['start'].tolist(), df["end"].tolist()

    anntags, annstarts, annends = get_ann_list(annpath)


    def get_tokens_tags():
        print('Extracting tokens & tags from ', txtpath)
        tokens = []
        tags = []
        raw_text = open(txtpath, 'r').read()
        start = 0
        for i in range(len(annstarts)):
            currtokens = nltk.word_tokenize(raw_text[start:annstarts[i]])
            if currtokens:
                currtags = ['O'] * len(currtokens) # O tagged entities
                tokens += currtokens
                tags += currtags

                currtokens = nltk.word_tokenize(raw_text[annstarts[i]:annends[i] + 1]) # B/I tagged entities
                currtags = ['B-' + anntags[i]] + ['I-' + anntags[i]] * (len(currtokens) - 1)
                tokens += currtokens
                tags += currtags

                start = annends[i] + 1

        currtokens = nltk.word_tokenize(raw_text[start:])
        currtags = ['O'] * len(currtokens)
        tokens += currtokens
        tags += currtags

        print('Extracted tokens & tags from ', txtpath)
        return tokens, tags

    tokens, tags = get_tokens_tags()

    def test():
        print('Running tests')
        assert (len(tokens) == len(tags))
        try:
            for j in range(1, len(tags)):
                if tags[j][0] == 'I' and tags[j - 1][0] == 'O':
                    raise Exception('I tag exists without a B tag at ', j)
        except Exception as e:
            print(e)
        print('Ran tests')

    test()

    # print("--------------- Training data ---------------")
    # for i in zip(tokens, tags):
    #     print(i)

    def create_csv():
        patientid = [patient] * len(tokens)
        try:
            with open(outpath, 'w', newline = '') as outfile:
                writer = csv.writer(outfile)
                writer.writerows(zip(patientid, tokens, tags))
                print('File saved - ', outpath)
        except Exception as e:
            print('Error while saving file - ', outpath, e)
            exit()

    create_csv()


for patient, txtpath, annpath, outpath in get_docs():
    print('Processing === ', patient)
    process(patient, txtpath, annpath, outpath)
    print('Processed === ', patient)
