# NOTES
# Remove relations
# Consider only VALID_TAGS
# Sort on start
# Multiple spans in one line(take end)
# Random characters in ann file like "" (102027, line 410)
import math
import os
import csv
import nltk
import pandas as pd

nltk.download('punkt')

ENV = 'train'
VER = 1 # v1 - default, v5 - sentence split, v8 - filtered overlap files + sentence split

SRC = {
    'train': 'track2-training_data_2',
    'val': 'track2-training_data_3',
    'test': 'gold-standard-test-data'
}

DEST = {
    'train': 'train',
    'val': 'val',
    'test': 'test'
}

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR, '..', 'data', SRC[ENV])
OUTPUT_DIR = os.path.join(CURR_DIR, '..', 'data', DEST[ENV])

SPAN_SIZE = 300
MAX_SEQ_UPPER = 350
MAX_SEQ_LOWER = 300
VALID_TAGS = ['Drug', 'Reason', 'ADE']
TAG_OVERLAP_FILES = {"101136", "100187", "129286", "106621", "108889", "110384", "122093", "110342", "111458", "113705", "178143", "113840", "110863", "102557", "110037", "119144", "111882", "100883", "114965", "117745", "109191", "113524", "113824", "116966", "105050", "123475", "110499", "192798", "100590", "101427", "103722", "100677", "103315", "112628", "115021", "107322", "134445", "100229", "114452", "105778", "103142", "115267", "105360", "111298", "109527", "116901", "185800", "114923", "114680", "110521", "103293", "107255", "103926", "141586", "105106", "109176", "119573", "115667", "157609", "107202", "108539", "107515", "115138", "106015", "109397", "121861", "181643", "104446", "110559", "126085", "112226", "121631", "106334", "123807", "106915", "116204", "105585", "189637", "115169", "119906", "111160", "103761", "103430", "106993", "107128", "105537", "146944", "113222", "106945", "115433", "105579", "109698", "104929", "142444", "112329", "109724"}

tagdict = {'B-Drug': 0,
          'I-Drug': 0,
          'B-Reason': 0,
          'I-Reason': 0,
          'B-ADE': 0,
          'I-ADE': 0,
          'O': 0,
          }

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

    # if os.path.exists(outpath):
    #     print('File already exists - ', outpath)
    #     return True

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

        df = df.loc[df["tag"].isin(VALID_TAGS)] # focus on selected entities

        df["end"] = df["end"].apply(lambda x: x.split(" ")[-1]) # take last pos of range(multiple spans when contd on next line)
        df['start'] = df['start'].astype(int) # string -> int
        df['end'] = df['end'].astype(int) # string -> int
        df.sort_values("start", inplace=True) # sort on start indices for pos match

        print('Extracted data from ', annpath)
        return df["tag"].tolist(), df['start'].tolist(), df["end"].tolist()

    anntags, annstarts, annends = get_ann_list(annpath)

    raw_text = open(txtpath, 'r').read()

    def get_tokens_tags():
        print('Extracting tokens & tags from ', txtpath)
        tokens = []
        tags = []
        start = 0
        for i in range(len(annstarts)):
            currtokens = nltk.word_tokenize(raw_text[start:annstarts[i]])
            if currtokens:
                currtags = ['O'] * len(currtokens) # O tagged entities
                tokens += currtokens
                tags += currtags

                currtokens = nltk.word_tokenize(raw_text[annstarts[i]:annends[i]]) # B/I tagged entities
                currtags = ['B-' + anntags[i]] + ['I-' + anntags[i]] * (len(currtokens) - 1)
                tokens += currtokens
                tags += currtags

                start = annends[i]

        currtokens = nltk.word_tokenize(raw_text[start:])
        currtags = ['O'] * len(currtokens)
        tokens += currtokens
        tags += currtags

        print('Extracted tokens & tags from ', txtpath)
        return tokens, tags

    tokens, tags = get_tokens_tags()

    for tag in tags:
        tagdict[tag] += 1

    """ v1: Arbitrary fixed length segmentation of txt"""
    def get_ids_v1():
        spanid = 1
        count = 0
        ids = []
        while count < len(tokens):
            currspan = min(len(tokens) - count, SPAN_SIZE)
            ids += ([spanid] * currspan)
            spanid += 1
            count += currspan

        return ids

    """ v5: Sentence level txt segmentation"""
    def get_ids_v5(tokens):
        sentences = nltk.sent_tokenize(raw_text)
        ids = []
        t = 0
        maxseqlen = 0
        maxlen = 0
        id = 0
        for sent in sentences:
            for w in nltk.word_tokenize(sent):
                if len(tokens[t]) < len(w):
                    sub = ""
                    while t < len(tokens):
                        sub += tokens[t]
                        t += 1
                        maxlen += 1

                        if sub == w:
                            break
                else:
                    if w == tokens[t] \
                            or (w == "``" and tokens[t] == "''") \
                            or (w == "''" and tokens[t] == "``") \
                            or tokens[t].startswith(w):
                        maxlen += 1
                        t += 1
                if maxlen >= MAX_SEQ_UPPER:
                    ids.extend([id for _ in range(maxlen)])
                    maxseqlen = max(maxseqlen, maxlen)
                    maxlen = 0
                    id += 1

            if MAX_SEQ_LOWER < maxlen < MAX_SEQ_UPPER:
                ids.extend([id for _ in range(maxlen)])
                maxseqlen = max(maxseqlen, maxlen)
                maxlen = 0
                id += 1

        if maxlen:
            ids.extend([id for _ in range(maxlen)])
            maxseqlen = max(maxseqlen, maxlen)

        return ids, maxseqlen

    ids, max_seq_len = get_ids_v5(tokens) if VER > 1 else (get_ids_v1(), SPAN_SIZE)

    def test():
        print('Running tests')
        assert (len(tokens) == len(tags))
        assert (len(tokens) == len(ids))
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
                writer.writerows(zip(patientid, ids, tokens, tags))
                print('File saved - ', outpath)
        except Exception as e:
            print('Error while saving file - ', outpath, e)
            exit()

    create_csv()

    return max_seq_len


max_seq_len_aggr = 0 # To calculate max_seq_len of the splits
for patient, txtpath, annpath, outpath in get_docs():
    print('Processing === ', patient)
    if VER == 8 and patient in TAG_OVERLAP_FILES:
        print('Skipped. Contains overlapping tags.')
        continue
    max_seq_len_aggr = max(max_seq_len_aggr, process(patient, txtpath, annpath, outpath))
    print('Processed === ', patient)

print(f'max_seq_len for {ENV} = {max_seq_len_aggr}')

total = 0
for k, v in tagdict.items():
    total += v
print('total - ', total)
for k, v in tagdict.items():
    score = math.log(total / float(tagdict[k]))
    score = score if score > 1.0 else 1.0
    print('weight of ' + k + ' is: ' + str(score))
