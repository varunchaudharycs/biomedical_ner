# TODO: Sampling, MetaMap Lite tokenizer

import os
import csv
import nltk
import pandas as pd
from collections import defaultdict

nltk.download('punkt')

ENV = 'val'
# 1 = simple sentence-level split
VER = 1

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


MAX_SEQ_UPPER = 150
MAX_SEQ_LOWER = 120
VALID_TAGS = ['Drug', 'Reason', 'ADE']


def get_docs():
    """
    Get a single patient data
    :return: patient's ID, text file, ann file and output CSV name
    """
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


def get_ann_list(annpath):
    """
    Get list of annotated tags and span
    :param annpath: annotation file
    :return: list of tags, starts and ends
    """
    # print('Extracting data from ', annpath)
    df = pd.DataFrame()
    try:
        df = pd.read_csv(annpath, delimiter='\t', names=['id', 'ann', 'entity'], doublequote=True)
    except Exception as e:
        print('Error while reading file - ', annpath, e)
        exit()
    # filter out R
    df = df[df['id'].str.startswith("T")]
    # split ann into tag-start-end
    df[["tag", "start", "end"]] = df["ann"].str.split(pat=' ', n=2, expand=True)
    # focus on selected entities
    df = df.loc[df["tag"].isin(VALID_TAGS)]
    # take last pos of range(multiple spans when contd on next line)
    df["end"] = df["end"].apply(lambda x: x.split(" ")[-1])
    # count each entity
    anntagcount = df.tag.value_counts()
    for valid_tag in VALID_TAGS:
        if valid_tag not in anntagcount:
            anntagcount[valid_tag] = 0
    # string -> int
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    # sort on start indices for pos match
    df.sort_values("start", inplace=True)

    tags, starts, ends, texts = df["tag"].tolist(), df['start'].tolist(), df["end"].tolist(), df["entity"].tolist()
    # print('Extracted data from ', annpath)

    return tags, starts, ends, texts, anntagcount


def filter_overlaps(tags, starts, ends, texts):
    """Filter a sequence of spans and remove duplicates or overlaps. Useful for
    creating named entities (where one token can only be part of one entity) or
    when merging spans with `Retokenizer.merge`. When spans overlap, the (first)
    longest span is preferred over shorter spans.
    spans (iterable): The spans to filter.
    RETURNS (list): The filtered spans.
    """
    spans = [{"tag": tag, "start": start, "end": end, "text": text} for tag, start, end, text in zip(tags, starts, ends, texts)]
    get_sort_key = lambda span: (span["end"] - span["start"], -span["start"])
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    filtered = defaultdict(int)
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span["start"] not in seen_tokens and span["end"] - 1 not in seen_tokens:
            result.append([span["tag"], span["start"], span["end"], span["text"]])
        else:
            filtered[span["tag"]] += 1
        seen_tokens.update(range(span["start"], span["end"]))

    result = sorted(result, key=lambda span: span[1])
    modtags, modstarts, modends, modtexts = zip(*result)
    return modtags, modstarts, modends, modtexts, filtered


def get_tokens_tags(rawtext, anntags, annstarts, annends, anntexts):
    """
    Get raw text tags and tokens
    :param rawtext: discharge summary
    :param anntags: tags from annotation file
    :param annstarts: start indices from annotation file
    :param annends: end indices from annotation file
    :return: tags and tokens from patient discharge summary after matching with ann file
    """
    # print('Extracting tokens & tags')
    tokens = []
    tags = []
    start = 0
    for i in range(len(annstarts)):
        currtokens = nltk.word_tokenize(rawtext[start:annstarts[i]])
        if currtokens:
            currtags = ['O'] * len(currtokens) # O tagged entities
            tokens.extend(currtokens)
            tags.extend(currtags)

        if rawtext[annstarts[i]:annends[i]].replace('\n', ' ') != anntexts[i]:
            print('Expected - ', anntexts[i])
            print('Actual - ', rawtext[annstarts[i]:annends[i]].replace('\n', ' '))
        currtokens = nltk.word_tokenize(rawtext[annstarts[i]:annends[i]].replace('\n', ' ')) # B/I tagged entities
        currtags = ['B-' + anntags[i]] + ['I-' + anntags[i]] * (len(currtokens) - 1)
        tokens.extend(currtokens)
        tags.extend(currtags)

        start = annends[i]

    currtokens = nltk.word_tokenize(rawtext[start:])
    currtags = ['O'] * len(currtokens)
    tokens.extend(currtokens)
    tags.extend(currtags)

    # print('Extracted tokens & tags')
    return tokens, tags


def get_sentence_ids(rawtext, tokens):
    sentences = nltk.sent_tokenize(rawtext)
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


def sanity_check(anntags, annstarts, annends, anncounts, filteredcounts, tokens, tags, ids, max_seq_len, patientid, tagcountdict):
    # print('Running tests')
    assert (len(tokens) == len(tags))
    assert (len(tokens) == len(ids))
    try:
        for j in range(1, len(tags)):
            if tags[j][0] == 'I' and tags[j - 1][0] == 'O':
                raise Exception('I tag exists without a B tag at ', j)
    except Exception as e:
        print(e)

    assert tagcountdict['B-Drug'] == anncounts['Drug'] - filteredcounts['Drug']
    assert tagcountdict['B-Reason'] == anncounts['Reason'] - filteredcounts['Reason']
    assert tagcountdict['B-ADE'] == anncounts['ADE'] - filteredcounts['ADE']

    # print('Ran tests')



def create_csv(patientid, ids, tokens, tags, outpath):
    try:
        with open(outpath, 'w', newline = '') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(zip(patientid, ids, tokens, tags))
            # print('File saved - ', outpath)
    except Exception as e:
        print('Error while saving file - ', outpath, e)
        exit()


def process(patient, txtpath, annpath, outpath):
    """
    Generate training data CSV for a patient
    :param patient: ID
    :param txtpath: discharge summary
    :param annpath: file containing tags and tokens
    :param outpath: train data CSV
    :return: saves CSV file, training data for given patient
    """

    anntags, annstarts, annends, anntexts, anncounts = get_ann_list(annpath)

    anntags, annstarts, annends, anntexts, filteredcounts = filter_overlaps(anntags, annstarts, annends, anntexts)

    rawtext = open(txtpath, 'r').read()

    tokens, tags = get_tokens_tags(rawtext, anntags, annstarts, annends, anntexts)

    ids, maxseqlen = get_sentence_ids(rawtext, tokens)

    patientid = [patient] * len(tokens)

    tagcountdict = defaultdict(int)
    for tag in tags:
        tagcountdict[tag] += 1

    sanity_check(anntags, annstarts, annends, anncounts, filteredcounts,
                 tokens, tags, ids, maxseqlen, patientid, tagcountdict)

    create_csv(patientid, ids, tokens, tags, outpath)

    return maxseqlen, anncounts, tagcountdict


if __name__ == "__main__":

    maxseqlen_aggr = 0  # To calculate max_seq_len of the splits
    originalentitycounts = defaultdict(int)  # in ann files
    finalentitycounts = defaultdict(int)  # after overlap removal
    finaltagcountdict = defaultdict(int)

    for patient, txtpath, annpath, outpath in get_docs():
        print('Processing === ', patient)
        maxseqlen, ogtagcountdict, tagcountdict = process(patient, txtpath, annpath, outpath)

        maxseqlen_aggr = max(maxseqlen_aggr, maxseqlen)
        for tag in VALID_TAGS:
            finalentitycounts[tag] += tagcountdict['B-' + tag]
            originalentitycounts[tag] += ogtagcountdict[tag]
        for k in tagcountdict:
            finaltagcountdict[k] += tagcountdict[k]
        # print('Processed === ', patient)

    print('ENV - ', ENV)
    print("Version - ", VER)
    print('Max Seq length LOWER - ', MAX_SEQ_LOWER)
    print('Max Seq length UPPER - ', MAX_SEQ_UPPER)
    print("Longest sentence size - ", maxseqlen_aggr)
    print('Original entities - ', originalentitycounts)
    print('Final entities - ', finalentitycounts)
    print('Final tags - ', finaltagcountdict)
