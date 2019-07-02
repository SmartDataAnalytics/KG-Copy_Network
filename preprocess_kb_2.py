import io
import os
import re
import json
import spacy
import unidecode
import numpy as np
from args import get_args
from fuzzywuzzy import process, fuzz
from spacy.tokenizer import Tokenizer
from multiprocessing import Pool, cpu_count
from spacy.lang.en.stop_words import STOP_WORDS
from collections import OrderedDict, defaultdict
from sklearn.metrics.pairwise import cosine_similarity


args = get_args()

STOP_WORDS.add('de_l_la_le_di')

#spacy tokenizers
nlp = spacy.load('en')
pos = spacy.load('en_core_web_lg')
tokenizer = Tokenizer(nlp.vocab)

#load word2index file
stoi = np.load(args.stoi,allow_pickle=True).item()
itos = {v: k for k, v in stoi.items()}
kg_club = 'data/KG/clubs/'
kg_country = 'data/KG/country/'

#output directories
out_dir = 'preproc_files/soccer/'
correct_pos = ['NOUN', 'PROPN', 'ADJ', 'NUM', 'VERB']
hit2team_maps = np.load('soccer_conversations/hit_team_maps.npy',allow_pickle=True).item()
team_kgs = {}
kg2idx_map = defaultdict(dict)
f=open('replaced.txt', 'w')
w_h_words = ['what', 'how', 'when', 'where', 'why', 'who']
blacklist = ['favorite','more','most', 'fastest', 'star' ,'remember', 'great' ,'famous', 'greatest','youngest','oldest', 'furthest', 'top', 'expensive','highest', 'best',
'average' ,'better', 'worst', 'avg', 'popular','cheapest','handsome', 'last',
'far', 'newest', 'founded', 'strongest']

vocab_glove = np.load(args.vocab_glove,allow_pickle=True).item()
global replaced
replaced = []
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
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\"", "", string)
    #string = re.sub(r"\.", " ", string)
    return ' ' +string.strip().lower()+ ' '


def generate_ngrams(s, n=[1, 2, 3, 4]):
    words_list = s.split()
    words_list = [w for w in words_list if w not in STOP_WORDS]
    ngrams_list = []

    for num in range(0, len(words_list)):
        for l in n:
            ngram = ' '.join(words_list[num:num + l])
            ngrams_list.append(ngram)
    return ngrams_list


def get_max_kb():
    kg_cl = os.listdir(kg_club)
    for kg_c in kg_cl:
        if kg_c:
            team_kgs[kg_c.replace('.txt', '')] = read_kg(kg_club+kg_c)
    kg_na = os.listdir(kg_country)
    for kg_n in kg_na:
        if kg_n:
            team_kgs[kg_n.replace('.txt', '')] = read_kg(kg_country+kg_n)

    max_len = np.max([(len(a)) for a, b, c in team_kgs.values()])
    return max_len

def duplicates(lst, item):
    """
    Get indices of duplicate elements in a list
    :param lst:
    :param item:
    :return:
    """
    return [idx for idx, x in enumerate(lst) if x == item]

def read_kg(file_n):
    """
    Get kg subject and relations
    :param file_n: input kg for team
    :return:question
    """
    with io.open(file_n, 'r', encoding='utf-8') as f:
        kg_info = f.readlines()
    #print (file_n)
    kg_info = [unidecode.unidecode(l) for l in kg_info]
    kg_sub = [info.replace('\n', '').split('\t')[0].strip().lower() for info in kg_info]
    kg_reln = [info.replace('\n', '').split('\t')[1].strip().lower() for info in kg_info]
    kg_obj = [info.replace('\n', '').split('\t')[-1].strip().lower() for info in kg_info]
    #print (kg_obj[0:10])
    return kg_sub, kg_reln, kg_obj


def check_question(question):
    question = ' '.join([itos[idx] for idx in question])
    if '?' in question:
        return True
    elif any(map(question.split()[0].__contains__, w_h_words)):
        return True
    else:
        return False


def get_avg_word2vec(phrase):
    """get word vectors for phrases"""
    vec = np.zeros(300)
    #print (phrase)
    phrase = phrase.strip()
    phrase = clean_str(phrase)
    #phrase = re.sub(r'\\','',phrase)
    #phrase = re.sub(r"\?", "", phrase)
    #phrase = re.sub(r"unk","",phrase)
    #phrase = re.sub(r"\)","",phrase)
    #phrase = re.sub(r"\(","",phrase)
    #print(phrase)
    for w in phrase.strip().split():
        #print ((vocab_glove[w]))
        try:
            vec = vec + np.array(vocab_glove[w]).reshape(1, 300).astype(np.float32)
        except KeyError:
            print("Phrase > ",phrase)
            vec = vec + np.array(vocab_glove[w]).reshape(1, 300).astype(np.float32)

            exit()
    return vec.reshape(1,300)

def get_rel_sim(relation, question):
    """
    Get max cosine distance for relations
    :param relation:
    :param question:
    """
    query_ngrams = generate_ngrams(question)
    query_ngrams_vec = [get_avg_word2vec(phr) for phr in query_ngrams]
    relation_ngram = get_avg_word2vec(relation)
    #print (relation_ngram)
    similarities = [cosine_similarity(relation_ngram, q)[0][0] for q in query_ngrams_vec]
    if similarities and np.max(similarities) > 0.5:
        return np.max(similarities)
    else:
        return 0.0


def get_fuzzy_match(object, answer, threshold=0.8):
    """get phrase with highest match in answer"""
    answer_phrase = generate_ngrams(answer)
    if answer_phrase:
        best_match = [fuzz.ratio(object, phr) for phr in answer_phrase]
        return np.max(best_match), answer_phrase[np.argmax(best_match)]
    else:
        return 0, ''



def check_presence(answer, kb_key):
    """check probable presence"""
    answer, match = process.extract(kb_key, answer)[0]
    if match > 0.5:
        return match
    else:
        return 0.0


def replace_obj(param):
    answer, team, question, dataset_type = param
    replaced_ans = ''
    if check_question(question) and dataset_type=='train':
        sub, rel, obj = team_kgs[team+'_kg']
        #check probable presence
        question = ' '.join([itos[idx] for idx in question])
        if any(x in question.split() for x in blacklist):
            # check if the question can be answered with the relation
            best_s = [(get_rel_sim(r, question), r) for r in rel]
            best_s = sorted(best_s, key=lambda x: x[0], reverse=True)
            #print (best_s[:3])
            if best_s[0][0] > 0.7: # probable relation present in kb
                # check if a probable object in the answer is present in the kb
                obj_presence = [(get_fuzzy_match(ob, answer), ob) for ob in obj]
                presence_score = [a[0] for a, b in obj_presence]
                print (question, answer)
                doc = pos(answer.strip())
                presence_pos = {}
                for p, o in enumerate(doc):
                    presence_pos[p] = o.pos_
                prob_presence = obj_presence[np.argmax(presence_score)][1].strip()
                prob_phrase = obj_presence[np.argmax(presence_score)][0][1].strip()
                if np.max(presence_score) < 60:
                    if len(prob_phrase.split()) < 2:
                        if prob_phrase in prob_presence:
                            presence_score = 70
                        else:
                            presence_score = np.max(presence_score)
                    else:
                        presence_score = np.max(presence_score)
                else:
                    presence_score = np.max(presence_score)
                presence_idx = [presence_pos[answer.strip().split().index(o)] for o in prob_phrase.split()]
                try:
                    prob_presence_val = int(prob_presence) > 10
                except ValueError:
                    prob_presence_val = True
                if presence_score > 60 and prob_presence_val and any(pos in presence_idx for pos in correct_pos): # presence in kb more than threshold
                    #prob_presence = obj_presence[np.argmax(presence_score)][1].strip()  # get best object
                    print (prob_presence)
                    dupl = duplicates(obj, prob_presence)  # duplicate indexes for repeated object
                    if len(dupl) > 1: # More than one probable object
                        print ("More than 1 match")
                        prob_rel = [(rel[d], d) for d in dupl]
                        print (prob_rel)
                        best_sim = [get_rel_sim(r, question) for r, d in prob_rel]  # get similarity with the adjacent relation and the query
                        print (best_sim)
                        if np.max(best_sim) > 0.7: # check if corresponding relation has high similarity with question.
                            print (prob_rel[np.argmax(best_sim)][1])
                            best_obj_idx = prob_rel[np.argmax(best_sim)][1]
                            best_obj = obj[best_obj_idx]
                        else:
                            best_obj = ''
                    else:
                        rel_sim = get_rel_sim(rel[dupl[0]], question)
                        if rel_sim > 0.7: # check if corresponding relation has high similarity with question.
                            best_obj = prob_presence
                            best_obj_idx = dupl[0]
                        else:
                            best_obj = ''


                    if best_obj:
                        replaced.append(best_obj_idx)
                        print ('Question was:' + question + ' and answer was:' + answer)
                        print ('Match found with match:' + str(prob_phrase) + ' with: ' + best_obj + ' with similarity='+ str(presence_score))
                        f.write('Question was:' + question + ' and answer was:' + answer + '\n')
                        f.write("Match found with match:" + str(prob_phrase) + " with: " + str(best_obj) + '\n')
                        #print (best_obj_idx[0])
                        replaced_ans = answer.replace(prob_phrase, 'o' + str(best_obj_idx))
                        f.write(answer + '\n')
                        f.write(replaced_ans + '\n')
                        replaced_ans = getsent2i(replaced_ans)
                        print ('Replaced Answer:' + str(replaced_ans))
                        f.write(str(replaced_ans) + '\n')
                        f.write('*' * 80 + '\n')
                        return  replaced_ans
                    else:
                        return getsent2i(answer)
                else:
                    return getsent2i(answer)
            else:
                return getsent2i(answer)
        else:
            return getsent2i(answer)
    else:
        return getsent2i(answer)


def get_chunks(query):
    chunks = np.zeros((len(query.split())))
    doc = pos(query)
    for e in doc.noun_chunks:
        chunks[e.start: e.end] = 1
    return chunks

def read_json(file_n):
    #read a json file
    json_f = file_n.split('/')[-1].replace('.json', '')
    team = hit2team_maps[json_f]
    if team:
        sub, reln, obj = team_kgs[team + '_kg']
        sub = [getsent2i(s) for s in sub]
        reln = [getsent2i(r) for r in reln]
        #obj_map = kg2idx_map[team + '_kg']
        #obj_map = [getw2id(o) for o in obj]
        with io.open(file_n, 'r', encoding='utf-8') as fp:
            #print (file_n)
            conv = json.load(fp, object_pairs_hook=OrderedDict,strict=False)
        q, q_c, a = [], [], []
        for k, v in conv.items():
            if 'q' in k:
                q.append(getsent2i(clean_str(v).strip()))
                q_c.append(get_chunks(clean_str(v).strip()))
            else:
                a.append(clean_str(v))

        if "train" in file_n:
            params = [(ans, team, q[j],"train") for j, ans in enumerate(a)]
        else:
            params = [(ans, team, q[j], "others") for j, ans in enumerate(a)]

        with Pool(processes=cpu_count()) as poo:
            answers_replaced = poo.map(func=replace_obj, iterable=params)
        print ("Number of replaced:" + str(len(replaced)))
        return q, q_c, answers_replaced, sub, reln, team+'_kg'
    else:
        with io.open(file_n, 'r', encoding='utf-8') as fp:
            #print (file_n)
            conv = json.load(fp, object_pairs_hook=OrderedDict,strict=False)
        q, a = [], []
        for k, v in conv.items():
            if 'q' in k:
                q.append(getsent2i(clean_str(v).strip()))
            else:
                a.append(getsent2i(clean_str(v).strip()))
        return q, [], a, [], [], ''


def getw2id(word):
    try:
        return stoi[word]
    except KeyError:
        return stoi['unk']


def getsent2i(sent):
    out = []
    sent = sent.strip()
    tokens = tokenizer(sent)
    for t in tokens:
        t = t.text
        out.append(getw2id(t))

    return out


def get_all_conv(dataset='train'):
    if dataset == 'val':
        print("---------------------------------------------------------------------VAL STARTED-------------------------------------------------",dataset)
        in_f = args.data_dir+'val/'
        dialogue_f = os.listdir(in_f)
        out_dial = [read_json(in_f + d_f) for d_f in dialogue_f]
        print("---------------------------------------------------------------------VAL DONE-------------------------------------------------")
    elif dataset == 'test':
        print("---------------------------------------------------------------------TEST STARTED-------------------------------------------------",dataset)
        in_f = args.data_dir+'test/'
        dialogue_f = os.listdir(in_f)
        out_dial = [read_json(in_f + d_f) for d_f in dialogue_f]
        print("---------------------------------------------------------------------TEST DONE-------------------------------------------------")
    elif dataset == 'train':
        print("---------------------------------------------------------------------TRAIN STARTED-------------------------------------------------",dataset)
        in_f = args.data_dir+'train/'
        dialogue_f = os.listdir(in_f)
        out_dial = [read_json(in_f + d_f) for d_f in dialogue_f]
        print("---------------------------------------------------------------------TRAIN DONE-------------------------------------------------")
    else:
        print("---------------------------------------------------------------------OTHERS STARTED-------------------------------------------------",dataset)
        files = os.listdir(args.data_dir)
        all_dial = []
        for data in files:
            in_f = args.data_dir + data
            if not os.path.isfile(in_f):
                all_dial.append([read_json(in_f + '/'+ d_f) for d_f in os.listdir(in_f)])
        out_dial = [dial for dialogues in all_dial for dial in dialogues]
        print("---------------------------------------------------------------------OTHERS DONE-------------------------------------------------")
    return out_dial


if __name__ == '__main__':
    max_kb_size = get_max_kb()
    print (max_kb_size)
    print (team_kgs.keys())
    outs = ['o'+str(i) for i in range(0, max_kb_size)]


    test = get_all_conv('test')
    np.save(out_dir+'test.npy', test)

    valid = get_all_conv('val')
    np.save(out_dir+'valid.npy', valid)

    train = get_all_conv()
    np.save(out_dir+'train.npy', train)

    print('Saving team KG')
    np.save(out_dir+'team_kg.npy', team_kgs)
    print('Saving the kg dictionary ')
    f.close()