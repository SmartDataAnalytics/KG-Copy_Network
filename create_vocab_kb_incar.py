import json
import numpy as np
import os
import spacy
from collections import defaultdict, OrderedDict
import unidecode
import re
from spacy.tokenizer import Tokenizer

# tokenizer
nlp = spacy.load('en')
tokenizer = Tokenizer(nlp.vocab)
# directiories
vocab_f = 'vocab/vocab_incar.npy'
vocab_glove_f = 'vocab/glove300_incar.npy'
w2id_f = 'vocab/w2i_incar.npy'
train_f = 'data/KVR/train_incar.txt'
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


def read_conversations(file_n):
    # read a soccer_conversations from the file
    queries = []
    answers = []
    with open(file_n, 'r', encoding='utf-8') as fp:
        for line in fp:
            if not ((line.startswith("0") or line.startswith("#")) and len(line)>0) and line!="\n":
                qas = line.strip().split("\t")
                queries.append(qas[0][2:len(qas[0])])
                answers.append(qas[1])
    return queries, answers


def read_kg(file_n):
    """
    Get kg subject and relations
    :param file_n: input kg for team
    :return:
    """
    kg_sub_reln = []
    with open(file_n, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("0"):
                triple = line.strip().split(" ")
                kg_sub_reln.append(triple[1]+" "+triple[2])
    return kg_sub_reln


def create_w2i(train_path):

    #Read all soccer_conversations
    all_sents = []
    ques, ans = read_conversations(train_path)
    for q in ques:
        all_sents.append(unidecode.unidecode(q))
    for a in ans:
        all_sents.append(unidecode.unidecode(a))

    # Adding all kg tokens
    kg_cl = read_kg(train_path)
    for s_r in kg_cl:
        all_sents.append(unidecode.unidecode(s_r))


    all_sents = [clean_str(sent) for sent in all_sents]

    print (len(all_sents))
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
    print ('created vocab dictionary with length:' + str(len(vocab)))
    np.save(vocab_f, vocab)
    unq_w = []
    unq_w.append('PAD')
    for w, c in vocab.items():
        if c > min_freq:
            unq_w.append(w)

    word2id = dict(zip(unq_w, range(0, len(unq_w))))
    print ('Total words in vocab: ' + str(len(word2id)))
    return vocab, word2id


if __name__ == '__main__':
    vocab, w2i = create_w2i(train_f)
    w2i['unk'] = len(w2i)+1
    w2i['<eos>'] = len(w2i)+1
    w2i['<sos>'] = len(w2i)+1
    vocab_glove = {}
    print ('Loading glove vectors..........')
    # load glove vectors
    with open(glove_pretrained, 'r', encoding='utf-8') as f:
        glove = f.readlines()
    words_found = []
    for line in glove:
        try:
            word = line.replace('\n','').split()[0]
            vec = line.replace('\n','').split()[-glove_dim:]
            if word in w2i.keys():
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
