# imports
import pickle
import numpy as np
from collections import defaultdict
import os
from collections import OrderedDict
import torch
from args import get_args
import json
import itertools
import re
args = get_args()
print ('Loaded all libraries')
class DialogBatcher:
    """
    Wrapper for batching the Soccer Dialogue dataset
    """
    def __init__(self, gpu=True, max_sent_len=100, max_resp_len=50, max_kb_len=198, max_kb_triple_len=6):
        self.batch_size = args.batch_size
        # self.use_mask = use_mask
        self.max_kb_len = max_kb_len
        self.max_kb_sub_len = max_kb_triple_len
        self.gpu = gpu
        self.max_sent_len = max_sent_len
        self.max_resp_len = args.resp_len
        if os.path.isfile(args.stoi):
            self.stoi = np.load(args.stoi,allow_pickle=True).item()

        self.vocab_glove = np.load(args.vocab_glove,allow_pickle=True).item()
        vec_dim = 300

        # self.stoi['EOS'] = len(self.stoi)+1
        # self.stoi['SOS'] = len(self.stoi)+1

        # Get required dictionaries for data
        self.train = self.get_sequences('train')
        self.test = self.get_sequences('test')
        self.valid = self.get_sequences('valid')
        #self.all = self.get_sequences('all')

        self.n_words = len(self.stoi) + 1
        self.n_train = len(self.train['x'])
        self.n_val = len(self.valid['x'])
        self.n_test = len(self.test['x'])
        #self.n_all = len(self.all)

        self.itos = {v : k for k, v in self.stoi.items()}

        # get pretrained vectors
        self.vectors = np.zeros((len(self.itos)+1, vec_dim))
        for k, v in self.vocab_glove.items():
            # self.vectors[self.stoi[k.encode('utf-8')]] = v
            self.vectors[self.stoi[k]] = v

        self.vectors = torch.from_numpy(self.vectors.astype(np.float32))


    def get_seq(self, dial, dataset):
        """
        get sequence prepared
        :param dial:
        :param dataset:
        :return:
        """
        out = []
        q, q_c, a, kb_s, kb_r, team = dial
        kb = [kb_s[j] + kb_r[j] for j in range(len(kb_s))]
        # dataset['kb'].append(kb)
        # dataset['team'].append(team)
        for l, (q_l, qc, a_l) in enumerate(zip(q, q_c, a)):
            dataset['q_c'].append(qc)
            dataset['kb'].append(kb)
            dataset['team'].append(team)
            dataset['y'].append(a_l+[self.stoi['<eos>']]) # add EOS token at the end

            if l > 0:
                out = self.merge_list([out, a[l - 1], q_l])
                dataset['x'].append(out + [self.stoi['<eos>']])
            else:
                out = self.merge_list([out, q_l])
                dataset['x'].append(out + [self.stoi['<eos>']])

    def get_sequences(self, file_name):
        """
        get dialogue data
        :param file_name:
        :return:
        """
        ds = {}
        ds['x'], ds['q_c'], ds['y'], ds['kb'], ds['team'] = [], [], [], [], []
        dat = np.load(args.data_dir+file_name+'.npy',allow_pickle=True)
        [self.get_seq(d, ds) for d in dat]
        ds['x'], ds['q_c'], ds['y'], ds['kb'] = np.array(ds['x']), np.array(ds['q_c']), np.array(ds['y']), np.array(ds['kb'])
        return ds
        #return dataset

    @staticmethod
    def merge_list(set_l):
        """
        merge previous utterances into current
        :param set_l:
        :return:
        """
        return list(itertools.chain.from_iterable(set_l))

    def geti2w(self, word):
        """
        get id 2 word
        :param word:
        :return:
        """
        if self.gpu:
            word = self.itos[int(word.cpu().numpy())]
            if isinstance(word, str):
                return word
            else:
                return word
        else:
            # word = self.itos[int(word.numpy())].decode('utf-8')
            word = self.itos[int(word.numpy())]
            if isinstance(word, str):
                return word
            else:
                return word

    def get_iter(self, dataset='train'):
        # get iterations.
        #self.batch_size = batch_size
        if dataset == 'train':
            dataset = self.train
        elif dataset == 'valid':
            dataset = self.valid
            # print(dataset['team'])
        elif dataset == 'test':
            dataset = self.test
        else:
            dataset = self.all

        for i in range(0, len(dataset['x']), self.batch_size):
            query = dataset['x'][i:i+self.batch_size]
            query_c = dataset['q_c'][i:i+self.batch_size]
            response = dataset['y'][i:i+self.batch_size]
            kb = dataset['kb'][i:i+self.batch_size]
            team = dataset['team'][i:i+self.batch_size]
        # for dat in dataset:
        #     query, response, kb, team = dat
            x, x_c, y, mx, my, kb, kb_m, s, v_m = self._load_batch(query, query_c, response, kb, self.batch_size)
        #
            yield x, x_c, y, mx, my, kb, kb_m, s, v_m, team

    def _load_batch(self, q, q_c, a, kb_i, b_s):
        b_s = min(b_s, len(q))
        #b_s = len(q)
        max_len_q = np.max([len(sent) for sent in q])
        max_len_q = (max_len_q) if max_len_q < self.max_sent_len else self.max_sent_len
        max_len_a = np.max([len(sent) for sent in a])
        max_len_a = (max_len_a) if max_len_a < self.max_resp_len else self.max_resp_len
        x = np.zeros([max_len_q, b_s], np.int)
        q_c_o = np.zeros([max_len_q, b_s], np.int)
        y = np.zeros([max_len_a, b_s], np.int)
        # sentient_g = np.zeros([max_len_a, b_s], np.int)
        kb = np.zeros([b_s, self.max_kb_len, self.max_kb_sub_len])
        kb_mask = np.zeros([b_s, self.max_kb_len])
        x_mask = np.zeros([max_len_q, b_s], np.int)
        y_mask = np.zeros([max_len_a, b_s], np.int)
        vocab_mask = np.arange(0, len(self.stoi) + 1)

        for j, (row_t, row_qc, row_l, row_kb) in enumerate(zip(q, q_c, a, kb_i)):
            row_t = row_t[-max_len_q:]
            row_qc = row_qc[-max_len_q:]
            row_l = row_l[:max_len_a]
            # print (kb_i)
            # print (row_t, len(row_t))
            x[:len(row_t), j] = row_t
            q_c_o[:len(row_qc), j] = row_qc
            y[:len(row_l), j] = row_l
            x_mask[:len(row_t), j] = 1
            y_mask[:len(row_l), j] = 1
            for l, k in enumerate(row_kb):
                try:
                    kb[j][l][:len(k)] = k
                except Exception:
                    print (k)
            kb_mask[j][:len(row_kb)] = 1

        x_o = torch.from_numpy(x)
        q_c_o = torch.from_numpy(q_c_o).type(torch.FloatTensor)
        y_o = torch.from_numpy(y).type(torch.FloatTensor)
        kb = torch.from_numpy(kb).long()
        sentient_g = (y_o > self.stoi['<eos>'])
        vocab_mask = torch.from_numpy(vocab_mask)
        vocab_mask = (vocab_mask < self.stoi['<eos>']).type(torch.FloatTensor)


        x_mask = torch.from_numpy(x_mask).type(torch.FloatTensor)
        y_mask = torch.from_numpy(y_mask).type(torch.FloatTensor)
        kb_mask = torch.from_numpy(kb_mask).type(torch.FloatTensor)

        if self.gpu:
            x_o, q_c_o, y_o, x_mask, y_mask, kb, kb_mask, sentient_g, vocab_mask = x_o.cuda(), q_c_o.cuda(), y_o.cuda(), x_mask.cuda(), y_mask.cuda(), \
                                                                kb.cuda(), kb_mask.cuda(), sentient_g.cuda(), vocab_mask.cuda()

        return x_o, q_c_o, y_o, x_mask, y_mask, kb, kb_mask, sentient_g.float(), vocab_mask


if __name__ == '__main__':
    batcher = DialogBatcher(gpu=False)
    batches = batcher.get_iter('valid')

    print (batches)

