from models.kg_copy_model import KGSentient
from args import get_args
import numpy as np
import torch
from batcher_kb_2 import DialogBatcher
from tqdm import tqdm
import pandas as pd
from utils_new import save_model, load_model
from metrics import EmbeddingMetrics
from bleu import get_moses_multi_bleu

# Get arguments
args = get_args()

# Set random seed
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

if args.gpu:
    torch.cuda.manual_seed(args.randseed)

#args.data_dir = 'preproc_files/'

# Team data
team_kg = np.load(args.data_dir+'team_kg.npy',allow_pickle=True).item()
# Get data
chat_data = DialogBatcher(gpu=args.gpu, max_kb_len=200, max_sent_len=20)
# Get model
model = KGSentient(hidden_size=args.hidden_size, max_r=args.resp_len, gpu=args.gpu, n_words=chat_data.n_words,
                       emb_dim=args.words_dim, kb_max_size=chat_data.max_kb_len, b_size=args.batch_size, lr=args.lr,
                       dropout=args.rnn_dropout, emb_drop=args.emb_drop, teacher_forcing_ratio=args.teacher_forcing,
                       pretrained_emb=chat_data.vectors, sos_tok=chat_data.stoi['<sos>'], eos_tok=chat_data.stoi['<eos>'],
                   itos=chat_data.geti2w, first_kg_token=chat_data.stoi['o0'])
metrics = EmbeddingMetrics(embeddig_dict=chat_data.vocab_glove)
model_name = 'Sentient_model2'
test_results = 'test_predicted_kg_attn2.csv'
test_out = pd.DataFrame()

w2i = np.load("vocab/w2i.npy",allow_pickle=True).item()
i2w = {v:k for k,v in w2i.items()}


ent_file = open("data/soccer_entities.txt","r",encoding="utf-8")

global_entity_list=set()

for entity in ent_file:
    global_entity_list.add(entity.strip())

global_entity_list = list(global_entity_list)


def toSentence(lst):
    list_tem = lst
    return " ".join(w2i[i] for i in list_tem).strip()

def processKG(filename):
    kg = ""
    try:
        kg = open('data/KG/clubs/'+filename+".txt", 'r', encoding="utf-8")
    except FileNotFoundError:
        kg = open('data/KG/country/' + filename + ".txt", 'r', encoding="utf-8")

    kg_all = []
    for line in kg:
        kg_all.append(['_'.join(a_line.split(" ")) for a_line in line.strip().split("\t")])
    return kg_all


def compute_prf(gold, pred, global_entity_list, kb_plain):
    local_kb_word = [k[0] for k in kb_plain]
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in set(pred):
            if p in global_entity_list or p in local_kb_word:
                if p not in gold:
                    FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    else:
        precision, recall, F1, count = 0, 0, 0, 0
    return F1, count


def compute_f1(gold,pred,entities,teams):
    epsilon = 0.000000001
    f1_score = 0.0
    microF1_TRUE = 0.0
    microF1_PRED = 0.0

    for it in range(len(gold)):
        f1_true, count = compute_prf(gold[it].split(),pred[it].split(),entities,processKG(teams[it]))
        microF1_TRUE += f1_true
        microF1_PRED += count

    f1_score = microF1_TRUE / float(microF1_PRED + epsilon)
    return f1_score


def train():

    best_val_loss = 100.0
    emb_val = -1000

    global_entity_list = []

    for epoch in range(args.epochs):
        epsilon = 0.000000001
        model.train()
        print('\n\n-------------------------------------------')
        print('Epoch-{}'.format(epoch))
        print('-------------------------------------------')

        train_iter = enumerate(chat_data.get_iter('train'))
        if not args.no_tqdm:
            train_iter = tqdm(train_iter)
            train_iter.set_description_str('Training')
            #print (chat_data.n_train)
            #train_iter.total = chat_data.n_train // chat_data.batch_size
            train_iter.total = chat_data.n_train // chat_data.batch_size
            #print (train_iter.total)
        for it, mb in train_iter:
            q, q_c, a, q_m, a_m, kb, kb_m, sentient, v_m, teams = mb
            model.train_batch(q, q_c, a, q_m, a_m, kb, kb_m, sentient)
            train_iter.set_description(model.print_loss())

        print('\n\n-------------------------------------------')
        print('Validation')
        print('-------------------------------------------')
        val_iter = enumerate(chat_data.get_iter('valid'))
        if not args.no_tqdm:
            val_iter = tqdm(val_iter)
            val_iter.set_description_str('Validation')
            val_iter.total = chat_data.n_val // chat_data.batch_size

        val_loss = 0.0
        extrema = []
        gm = []
        emb_avg_all = []
        predicted_s = []
        orig_s = []
        f1_score = 0.0
        for it, mb in val_iter:
            q, q_c, a, q_m, a_m, kb, kb_m, sentient, v_m, teams = mb
            pred, loss = model.evaluate_batch(q, q_c, a, q_m, a_m, kb, kb_m, sentient)

            # print ('=================Predicted vectors================')
            # print (pred[0])
            pred = pred.transpose(0, 1).contiguous()
            a = a.transpose(0, 1).contiguous()
            #print (a.size())
            s_g = get_sentences(a, teams)
            s_p = get_sentences(pred, teams)
            e_a, v_e, g_m = metrics.get_metrics_batch(s_g, s_p)
            f1_score += compute_f1(s_g,s_p,global_entity_list,teams)

            emb_avg_all.append(e_a)
            extrema.append(v_e)
            gm.append(g_m)
            predicted_s.append(s_p)
            orig_s.append(s_g)
            val_loss += loss.item()

        print('\n\n-------------------------------------------')
        print ('Sample prediction')
        print('-------------------------------------------')
        #rand = random.randint(0, )
        #p = s_p[rand]
        #o = s_g[rand]
        print (str(a))
        for k, o in enumerate(s_g):
            print ('Original:' + o)
            try:
                print ('Predicted:' + s_p[k])
            except UnicodeEncodeError:
                print ('Predicted: '.format(s_p[k]))
        print('-------------------------------------------')
        #v_l = val_loss/val_iter.total
        #ea = np.average(extrema)
        print("Vector extrema:" + str(np.average(extrema)))
        print("Greedy Matching:" + str(np.average(gm)))
        print('Embedding Average for this epoch:{:.6f}'.format(np.average(emb_avg_all)))
        predicted_s = [q for ques in predicted_s for q in ques]
        orig_s = [q for ques in orig_s for q in ques]
        moses_bleu = get_moses_multi_bleu(predicted_s, orig_s, lowercase=True)
        print ('Length of pred:' + str(len(orig_s)) + ' moses bleu: '+ str(moses_bleu))
        f1 = f1_score/len(val_iter)
        print("F1 score: ", f1)
        #ea = moses_bleu
        if f1 > emb_val:
            emb_val = f1
            print ('Saving best model')
            save_model(model, model_name)
        else:
            print ('Not saving the model. Best validation moses bleu so far:{:.4f}'.format(emb_val))
        print ('Validation Loss:{:.2f}'.format(val_loss/val_iter.total))
        # sprint ('Embedding average:{:.6f}'.format(emb_val))


def test(model):
    model = load_model(model, model_name, args.gpu)
    print('\n\n-------------------------------------------')
    print('Testing')
    print('-------------------------------------------')
    test_iter = enumerate(chat_data.get_iter('test'))
    if not args.no_tqdm:
        test_iter = tqdm(test_iter)
        test_iter.set_description_str('Testing')
        test_iter.total = chat_data.n_test // chat_data.batch_size

    test_loss = 0.0
    extrema = []
    gm = []
    emb_avg_all = []
    predicted_s = []
    orig_s = []
    f1_score = 0.0
    for it, mb in test_iter:
        q, q_c, a, q_m, a_m, kb, kb_m, sentient, v_m, teams = mb
        pred, loss = model.evaluate_batch(q, q_c, a, q_m, a_m, kb, kb_m, sentient)
        # print('=================Predicted vectors================')
        # print(pred[0])
        pred = pred.transpose(0, 1).contiguous()
        a = a.transpose(0, 1).contiguous()
        s_g = get_sentences(a, teams)
        s_p = get_sentences(pred, teams)
        e_a, v_e, g_m = metrics.get_metrics_batch(s_g, s_p)
        f1_score += compute_f1(s_g, s_p, global_entity_list, teams)
        emb_avg_all.append(e_a)
        extrema.append(v_e)
        gm.append(g_m)
        predicted_s.append(s_p)
        orig_s.append(s_g)
        test_loss += loss.item()



    print ("Vector extrema:" + str(np.average(extrema)))
    print ("Greedy Matching:" + str(np.average(gm)))
    print ("Embedding Average on Test:{:.6f}".format(np.average(emb_avg_all)))
    print('\n\n-------------------------------------------')
    print ('Sample prediction Test')
    print('-------------------------------------------')
    print ('words:' + str(a))
    for k, o in enumerate(s_g):
        print ('Original:' + o)
        try:
            print ('Predicted:' + s_p[k])
        except UnicodeEncodeError:
            print ('Predicted: '.format(s_p[k]))
    print('-------------------------------------------')
    print('-------------------------------------------')
    predicted_s = [q for ques in predicted_s for q in ques]
    orig_s = [q for ques in orig_s for q in ques]
    moses_bleu = get_moses_multi_bleu(predicted_s, orig_s, lowercase=True)
    print ("Moses Bleu:" + str(moses_bleu))
    print("F1 score: ", f1_score / len(test_iter))
    test_out['original_response'] = orig_s
    test_out['predicted_response'] = predicted_s
    print ('Saving the test predictions......')
    test_out.to_csv(test_results, index=False)


def get_sentences(sent_indexed, teams):
    # print (len(teams), len(sent_indexed))
    #print (sent_indexed)
    out_sents = [get_sent(sent_indexed[i], teams[i]) for i in range(len(sent_indexed))]
    # print (out_sents[0])
    # out_sents = [sent for sent in out_sents]
    #fetched =
    #fetched = [fetched for sent, fetched in out_sents]
    out_sents = [str(sent.split('<eos>')[0]) for sent, fetched in out_sents]

    #print ('Fetched from KB:' + str(np.sum(out_sents[:, 1])))
    return out_sents


def get_sent(sent, team):
    team_o = {}
    fetched_from_kb = 0
    number152 = ""
    if team:
        # print (team)
        s, r, o = team_kg[team]
        for i, obj in enumerate(o):
            team_o['o' + str(i)] = obj
            if i==152:
                number152 = obj
    out_sent = []
    for idx in sent:
        w = chat_data.geti2w(idx)
        #print (w)
        if team_o:
            try:
                #print ('Fetching from KG....')
                out_sent.append(team_o[w])
                fetched_from_kb=1
            except KeyError:
                out_sent.append(w)
                #print("w ------> ",w)
                #print("team ----> ",team)
                #print("kg 152 -----> ",number152)
        else:
            out_sent.append(w)

    return ' '.join(out_sent), fetched_from_kb


if __name__ == '__main__':
    train()
    test(model)
