import json
import numpy as np
import os
import io
import spacy
from collections import defaultdict, OrderedDict
import unidecode
import re
from spacy.tokenizer import Tokenizer

# tokenizer
nlp = spacy.load('en')
tokenizer = Tokenizer(nlp.vocab)
# directiories
vocab_f = 'vocab/vocab.npy'
vocab_glove_f = 'vocab/glove300.npy'
w2id_f = 'vocab/w2i.npy'
train = 'soccer_conversations/train/'
kg_club = 'data/KG/clubs/'
kg_country = 'data/KG/country/'
glove_pretrained = 'vocab/wiki.en.vec'
glove_dim = 300
min_freq = 1


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = unidecode.unidecode(string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\"", "", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # string = re.sub(r"\.", " ", string)
    return string.strip().lower()


def read_json(file_n):
    # read a json file
    with io.open(file_n, 'r', encoding='utf-8') as fp:
        # print (file_n)
        conv = json.load(fp, object_pairs_hook=OrderedDict,strict=False)
    queries = []
    answers = []
    for k, v in conv.items():
        if 'q' in k:
            queries.append(v)
        else:
            answers.append(v)

    return queries, answers


def read_kg(file_n):
    """
    Get kg subject and relations
    :param file_n: input kg for team
    :return:
    """
    with io.open(file_n, 'r', encoding='utf-8') as f:
        kg_info = f.readlines()
    kg_sub_reln = [re.sub("\s\s+" , " ", ' '.join(info.replace('\n', '').split('\t')[0:2])) for info in kg_info]

    return kg_sub_reln


def create_w2i(train_path):
    # create word2id
    if os.path.exists(vocab_f):
        vocab = np.load(vocab_f,allow_pickle=True).item()
    else:
        all_sents = []
        train_files = os.listdir(train_path)
        for tr_f in train_files:
            ques, ans = read_json(train + tr_f)
            for q in ques:
                all_sents.append(unidecode.unidecode(q))
            for a in ans:
                all_sents.append(unidecode.unidecode(a))

        # Adding all kg tokens
        all_kgs_club = os.listdir(kg_club)
        for kb in all_kgs_club:
            kg_cl = read_kg(kg_club+kb)
            for s_r in kg_cl:
                #print(s_r)
                all_sents.append(unidecode.unidecode(s_r))
        all_kgs_nation = os.listdir(kg_country)
        for kb in all_kgs_nation:
            kg_na = read_kg(kg_country+kb)
            for s_r in kg_na:
                all_sents.append(unidecode.unidecode(s_r))

        all_sents = [clean_str(sent) for sent in all_sents]
        print (len(all_sents))
        print (len(train_files))
        # print ('# utterances/dialogue: ' + str(float(len(all_sents)) / len(train_files)))
        # create vocab
        print ('Creating the vocabulary.....')
        vocab = defaultdict(float)
        for sent in all_sents:
            tokens = tokenizer(sent)
            for token in tokens:
                # print (token)
                token = token.text
                if token:
                    if isinstance(token, str):
                        vocab[token.lower()] += 1.0
                    else:
                        vocab[token] += 1.0
                # vocab[token.lower() if isinstance(token, str) else token] += 1.0
        print ('created vocab dictionary with length:' + str(len(vocab)))
    # print (vocab)
    np.save(vocab_f, vocab)
    unq_w = []
    unq_w.append('PAD')
    for w, c in vocab.items():
        # print (w, c)
        if c > min_freq:
            unq_w.append(w)

    word2id = dict(zip(unq_w, range(0, len(unq_w))))
    print ('Total words in vocab: ' + str(len(word2id)))
    return vocab, word2id


if __name__ == '__main__':
    vocab, w2i = create_w2i(train)
    if "(" in list(vocab.keys()) or "(" in list(w2i.keys()):
        print("(")
    w2i['unk'] = len(w2i)+1
    w2i['<eos>'] = len(w2i)+1
    w2i['<sos>'] = len(w2i)+1
    vocab_glove = {}
    print ('Loading glove vectors..........')
    # load glove vectors
    with io.open(glove_pretrained, 'r', encoding='utf-8') as f:
        glove = f.readlines()
    words_found = []
    for line in glove:
        try:
            word = line.replace('\n','').split()[0]
            vec = line.replace('\n','').split()[-glove_dim:]
            if word in w2i.keys():
                # print (word)
                words_found.append(word)
                vocab_glove[word] = np.array(vec)
            else:
                continue
        except Exception:
            continue
    for word in w2i.keys():
        if word not in words_found:
            vocab_glove[word] = np.random.uniform(-0.25, 0.25, glove_dim)
    for i in range(0, 200):
        w2i['o'+str(i)] = len(w2i) + 1
    np.save(w2id_f, w2i)
    np.save(vocab_glove_f, vocab_glove)


