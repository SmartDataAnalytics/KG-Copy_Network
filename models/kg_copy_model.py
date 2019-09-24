#imports
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import optim
from torch.autograd import Variable
#from utils import masked_cross_entropy
from utils_new import masked_cross_entropy

class KGSentient(nn.Module):
    """
    Sequence to sequence model with Attention
    """
    def __init__(self, hidden_size, max_r, n_words, b_size, emb_dim, sos_tok, eos_tok, itos, kb_max_size, gpu=False,
                 lr=0.001, train_emb=False, n_layers=1, clip=4.0, pretrained_emb=None, dropout=0.1, emb_drop=0.5,
                 teacher_forcing_ratio=5.0, sent_loss_ratio = 0.02, first_kg_token=4740):
        super(KGSentient, self).__init__()
        self.name = "VanillaSeq2Seq"
        self.input_size = n_words
        self.output_size = n_words
        self.hidden_size = hidden_size
        self.max_r = max_r ## max response len
        self.lr = lr
        self.emb_dim = emb_dim
        self.decoder_learning_ratio = 5.0
        self.kb_max_size = kb_max_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.emb_drop = emb_drop
        self.b_size = b_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.sos_tok = sos_tok
        self.eos_tok = eos_tok
        self.itos = itos
        self.clip = clip
        self.use_cuda = gpu
        self.sentient_loss = nn.BCELoss()
        # Common embedding for both encoder and decoder
        #self.embedding = nn.Embedding(self.output_size, self.emb_dim, padding_idx=0)
        #if pretrained_emb is not None:
        #    self.embedding.weight.data.copy_(pretrained_emb)
        #if train_emb == False:
        #    self.embedding.weight.requires_grad = False


        # Indexes for output vocabulary
        if self.use_cuda:
            self.kg_vocab = torch.from_numpy(np.arange(first_kg_token, first_kg_token+self.kb_max_size)).long().cuda()
        else:
            self.kg_vocab = torch.from_numpy(np.arange(first_kg_token, first_kg_token+self.kb_max_size)).long()

        # Use single RNN for both encoder and decoder
        #self.rnn = nn.LSTM(emb_dim, hidden_size, n_layers, dropout=dropout)
        # initializing the model
        self.encoder = EncoderRNN(self.n_layers, self.emb_dim, self.hidden_size, self.b_size, self.output_size,
                                 gpu=self.use_cuda, pretrained_emb=pretrained_emb)
        self.decoder = Decoder(self.hidden_size, self.emb_dim, self.output_size)

        self.sentinel_g = SentientAttention(self.kb_max_size, self.hidden_size, self.output_size, self.emb_dim,
                                            pretrained_emb, self.use_cuda)

        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            #self.embedding = self.embedding.cuda()
            self.sentinel_g = self.sentinel_g.cuda()
            #self.rnn = self.rnn.cuda()

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr * self.decoder_learning_ratio)

        self.loss = 0
        self.print_every = 1

    def train_batch(self, input_batch, input_chunk, out_batch, input_mask, target_mask, kb, kb_mask, sentient_orig):

        self.encoder.train(True)
        self.decoder.train(True)
        #self.embedding.train(True)

        #inp_emb = self.embedding(input_batch)
        #print (len(out_batch))
        b_size = input_batch.size(1)
        #print (b_size)
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss_Vocab,loss_Ptr,loss_Gate = 0,0,0
        # Run words through encoder
        #input_len = torch.sum(input_mask, dim=0)
        encoder_outputs, encoder_hidden = self.encoder(input_batch, input_mask)

        #target_len = torch.sum(target_mask, dim=0)
        target_len = out_batch.size(0)
        #print (min(max(target_len), self.max_r))
        max_target_length = min(target_len, self.max_r)
        #print (max_target_length)
        if not isinstance(max_target_length, int):
            max_target_length = int(max_target_length.cpu().numpy()) if self.use_cuda else int(max_target_length.numpy())

        # Prepare input and output variables
        if self.use_cuda:
            decoder_input = Variable(torch.Tensor([self.sos_tok] * b_size)).cuda().long()
            sentinel_values = Variable(torch.zeros(int(max_target_length), b_size)).cuda()
            all_decoder_outputs_vocab = Variable(torch.zeros(int(max_target_length), b_size, self.output_size)).cuda()
        else:
            decoder_input = Variable(torch.Tensor([self.sos_tok] * b_size)).long()
            sentinel_values = Variable(torch.zeros(int(max_target_length), b_size))
            all_decoder_outputs_vocab = Variable(torch.zeros(int(max_target_length), b_size, self.output_size))

        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers], encoder_hidden[1][:self.decoder.n_layers])

        # Choose whether to use teacher forcing
        use_teacher_forcing = random.randint(0, 10) < self.teacher_forcing_ratio

        if use_teacher_forcing:
            for t in range(max_target_length):
                #inp_emb_d = self.embedding(decoder_input)
                #print (decoder_input.size())
                decoder_vocab, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs, input_mask)
                # sentient gating
                sentient_gate, obj = self.sentinel_g(input_batch, input_chunk, input_mask, decoder_input,
                                                     kb, kb_mask, sentinel_values[t - 1])
                #s = sentient_orig[t].reshape(b_size, 1)
                s = F.sigmoid(sentient_gate)
                obj = s * obj
                decoder_vocab = (1 - s) * decoder_vocab
                decoder_vocab = decoder_vocab.scatter_add(1, self.kg_vocab.repeat(b_size).view(b_size, self.kb_max_size), obj)
                sentinel_values[t] = F.sigmoid(sentient_gate).squeeze()
                all_decoder_outputs_vocab[t] = decoder_vocab
                decoder_input = out_batch[t].long() # Next input is current target
        else:
            print ('Not TF..')
            for t in range(max_target_length):
                #inp_emb_d = self.embedding(decoder_input)
                decoder_vocab, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs, input_mask)
                all_decoder_outputs_vocab[t] = decoder_vocab
                sentient_gate, obj = self.sentinel_g(input_batch, input_chunk, input_mask, decoder_input,
                                                     kb, kb_mask, sentinel_values[t - 1])
                s = F.sigmoid(sentient_gate)
                sentinel_values[t] = s.squeeze()
                obj = s * obj
                decoder_vocab = (1 - s) * decoder_vocab
                decoder_vocab = decoder_vocab.scatter_add(1, self.kg_vocab.repeat(b_size).view(b_size, 200), obj)
                all_decoder_outputs_vocab[t] = decoder_vocab
                topv, topi = decoder_vocab.data.topk(1) # get prediction from decoder
                decoder_input = Variable(topi.view(-1)) # use this in the next time-steps

        #print (all_decoder_outputs_vocab.size(), out_batch.size())
        #out_batch = out_batch.transpose(0, 1).contiguous
        target_mask = target_mask.transpose(0,1).contiguous()
        #print (all_decoder_outputs_vocab.size(), out_batch.size(), target_mask.size())
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(), # -> B x S X VOCAB
            out_batch.transpose(0, 1).contiguous(), # -> B x S
            target_mask
        )
        sentiental_loss = self.sentient_loss(sentinel_values, sentient_orig)

        loss = loss_Vocab + sentiental_loss
        loss.backward()

        # clip gradient
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)


        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.item()


    def evaluate_batch(self, input_batch, input_chunk, out_batch, input_mask, target_mask, kb, kb_mask, sentient_orig):
        """
        evaluating batch
        :param input_batch:
        :param out_batch:
        :param input_mask:
        :param target_mask:
        :return:
        """
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        #self.embedding.train(False)

        #inp_emb = self.embedding(input_batch)
        # output decoder words



        encoder_outputs, encoder_hidden = self.encoder(input_batch, input_mask)
        b_size = input_batch.size(1)
        #target_len = torch.sum(target_mask, dim=0)
        target_len = out_batch.size(0)
        #print (min(max(target_len), self.max_r))
        max_target_length = (min(target_len, self.max_r))
        #print (max_target_length)
        if not isinstance(max_target_length, int):
            max_target_length = int(max_target_length.cpu().numpy()) if self.use_cuda else int(max_target_length.numpy())


        # Prepare input and output variables
        if self.use_cuda:
            decoder_input = Variable(torch.Tensor([self.sos_tok] * b_size)).long().cuda()
            sentinel_values = Variable(torch.zeros(int(max_target_length), b_size)).cuda()
            all_decoder_outputs_vocab = Variable(torch.zeros(int(max_target_length), b_size, self.output_size)).cuda()
        else:
            decoder_input = Variable(torch.Tensor([self.sos_tok] * b_size)).long()
            sentinel_values = Variable(torch.zeros(int(max_target_length), b_size))
            all_decoder_outputs_vocab = Variable(torch.zeros(int(max_target_length), b_size, self.output_size))

        decoded_words = Variable(torch.zeros(int(max_target_length), b_size)).cuda() if self.use_cuda else \
                        Variable(torch.zeros(int(max_target_length), b_size))
        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers], encoder_hidden[1][:self.decoder.n_layers])
        # provide data to decoder
        for t in range(max_target_length):
            #print (decoder_input)
            #inp_emb_d = self.embedding(decoder_input)
            decoder_vocab, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs, input_mask)
            sentient_gate, obj = self.sentinel_g(input_batch, input_chunk, input_mask, decoder_input,
                                                 kb, kb_mask, sentinel_values[t - 1])
            # print (sentient_gate.size())
            # obj_output = (torch.cat([vocab_pad, obj], dim=-1))
            s = F.sigmoid(sentient_gate)
            sentinel_values[t] = s.squeeze()
            obj = s * obj
            decoder_vocab = (1 - s) * decoder_vocab
            decoder_vocab = decoder_vocab.scatter_add(1, self.kg_vocab.repeat(b_size).view(b_size, self.kb_max_size), obj)
            all_decoder_outputs_vocab[t] = decoder_vocab
            topv, topi = decoder_vocab.data.topk(1) # get prediction from decoder
            decoder_input = Variable(topi.view(-1)) # use this in the next time-steps
            decoded_words[t] = (topi.view(-1))

        target_mask = target_mask.transpose(0,1).contiguous()

        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(), # -> B x S X VOCAB
            out_batch.transpose(0, 1).contiguous(), # -> B x S
            target_mask
        )

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        #self.embedding.train(True)

        return decoded_words, loss_Vocab

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        self.print_every += 1
        return 'L:{:.2f}'.format(print_loss_avg)

    def get_single_resp(self, input_batch, input_chunk, input_mask, target_len, kb, kb_mask):
        # Get Response for a single query
        self.encoder.train(False)
        self.decoder.train(False)

        max_target_length = target_len
        b_size = input_batch.size(1)


        # Prepare input and output variables
        if self.use_cuda:
            input_batch, input_chunk, input_mask = input_batch.cuda().long(), input_chunk.cuda(), input_mask.cuda()
            kb, kb_mask = kb.cuda().long(), kb_mask.cuda()
            decoder_input = Variable(torch.Tensor([self.sos_tok] * b_size)).long().cuda()
            sentinel_values = Variable(torch.zeros(int(max_target_length), b_size)).cuda()
            all_decoder_outputs_vocab = Variable(torch.zeros(int(max_target_length), b_size, self.output_size)).cuda()
        else:
            input_batch, input_chunk, input_mask = input_batch.long(), input_chunk.long(), input_mask
            kb, kb_mask = kb.long(), kb_mask.long()
            decoder_input = Variable(torch.Tensor([self.sos_tok] * b_size)).long()
            sentinel_values = Variable(torch.zeros(int(max_target_length), b_size))
            all_decoder_outputs_vocab = Variable(torch.zeros(int(max_target_length), b_size, self.output_size))

        encoder_outputs, encoder_hidden = self.encoder(input_batch, input_mask)

        decoded_words = Variable(torch.zeros(int(max_target_length), b_size)).cuda() if self.use_cuda else \
                        Variable(torch.zeros(int(max_target_length), b_size))
        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers], encoder_hidden[1][:self.decoder.n_layers])

        for t in range(max_target_length):
            #print (decoder_input)
            #inp_emb_d = self.embedding(decoder_input)
            decoder_vocab, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs, input_mask)
            sentient_gate, obj = self.sentinel_g(input_batch, input_chunk, input_mask, decoder_input,
                                                 kb, kb_mask, sentinel_values[t - 1])
            # print (sentient_gate.size())
            # obj_output = (torch.cat([vocab_pad, obj], dim=-1))
            s = F.sigmoid(sentient_gate)
            sentinel_values[t] = s.squeeze()
            obj = s * obj
            decoder_vocab = (1 - s) * decoder_vocab
            decoder_vocab = decoder_vocab.scatter_add(1, self.kg_vocab.repeat(b_size).view(b_size, 200), obj)
            all_decoder_outputs_vocab[t] = decoder_vocab
            topv, topi = decoder_vocab.data.topk(1) # get prediction from decoder
            decoder_input = Variable(topi.view(-1)) # use this in the next time-steps
            decoded_words[t] = (topi.view(-1))

        return decoded_words

class EncoderRNN(nn.Module):
    """
    Encoder RNN module
    """
    def __init__(self, input_size, emb_size, hidden_size, b_size, vocab_size, n_layers=1, dropout=0.1, emb_drop=0.2,
                 gpu=False, pretrained_emb=None, train_emb=True):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_cuda = gpu
        self.b_size = b_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.gpu = gpu
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.embedding_dropout = nn.Dropout(emb_drop)
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, dropout=self.dropout)
        if pretrained_emb is not None:
           self.embedding.weight.data.copy_(pretrained_emb)
        if train_emb == False:
           self.embedding.weight.requires_grad = False
        #self.rnn = rnn

    def init_weights(self, b_size):
        #intiialize hidden weights
        c0 = Variable(torch.zeros(self.n_layers, b_size, self.hidden_size))
        h0 = Variable(torch.zeros(self.n_layers, b_size, self.hidden_size))

        if self.gpu:
            c0 = c0.cuda()
            h0 = h0.cuda()

        return h0, c0

    def forward(self, inp_q, input_mask, input_lengths=None):
        #input_q =numpy S X B input_mask = S X B
        #embeddeinputs, inp_mask, encoder_hidden, decoder_out, kb, kb_mask, last_sentientd = self.embedding(input_q)

        embedded = self.embedding_dropout(self.embedding(inp_q)) # S X B X E
        hidden = self.init_weights(embedded.size(1))

        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)

        outputs, hidden = self.rnn(embedded, hidden) # outputs = S X B X n_layers*H, hidden = 2 * [1 X B X H]
        outputs = outputs * input_mask.unsqueeze(-1)
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        return outputs, hidden

class Attention(nn.Module):
    """
    Attention mechanism (Luong)
    """
    def __init__(self, hidden_size, hidden_size1):
        super(Attention, self).__init__()
        #weights
        self.W_h = nn.Linear(hidden_size + hidden_size1, hidden_size, bias=False)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.epsilon = 1e-10

    def forward(self, encoder_outputs, decoder_hidden, inp_mask):
        seq_len = encoder_outputs.size(1) # get sequence lengths S
        H = decoder_hidden.repeat(seq_len, 1, 1).transpose(0, 1) # B X S X H
        energy = F.tanh(self.W_h(torch.cat([H, encoder_outputs], 2))) # B X S X H
        energy = energy.transpose(2, 1)
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B X 1 X H]
        energy = torch.bmm(v,energy).view(-1, seq_len) # [B X T]
        a = F.softmax(energy, dim=-1) * inp_mask.transpose(0, 1) # B X T
        normalization_factor = a.sum(1, keepdim=True)
        a = a / (normalization_factor+self.epsilon) # adding a small offset to avoid nan values

        a = a.unsqueeze(1)
        context = a.bmm(encoder_outputs)

        return a, context

class Decoder(nn.Module):
    """
    Decoder RNN
    """
    def __init__(self, hidden_size, emb_dim, vocab_size, n_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hidden_size, n_layers, dropout=dropout)
        #self.rnn = rnn
        self.out = nn.Linear(self.hidden_size, vocab_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)

        # Attention
        self.attention = Attention(hidden_size, hidden_size)

    def forward(self, input_q, last_hidden, encoder_outputs, inp_mask):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        inp_emb = self.embedding(input_q)
        #print (inp_emb.size())
        batch_size = min(last_hidden[0].size(1), inp_emb.size(0))
        inp_emb = inp_emb[-batch_size:]

        #max_len = encoder_outputs.size(0)args
        encoder_outputs = encoder_outputs.transpose(0,1) # B X S X H
        #embedded = self.embedding(input_seq)
        embedded = self.dropout(inp_emb)
        #print (embedded.size())
        embedded = embedded.view(1, batch_size, self.emb_dim) # S=1 x B x N
        #print (embedded.size())
        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.rnn(embedded, last_hidden)

        s_t = hidden[0][-1].unsqueeze(0)
        #s_t = (rnn_output * input_q_mask.unsqueeze(-1))[-1].squeeze()

        alpha, context = self.attention(encoder_outputs, s_t, inp_mask)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x H -> B x H
        context = context.squeeze(1)       # B x S=1 x H -> B x H
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))
        #print (concat_output.size())
        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden

class SentientAttention(nn.Module):
    """
    Sentinel Module
    """
    def __init__(self, kb_max_size, hidden_size, vocab_size, emb_dim, pretrained_emb=None, cuda=False):
        super(SentientAttention, self).__init__()
        self.kb_size = kb_max_size
        self.sentinel_gate = nn.Linear(emb_dim + self.kb_size + 1, 1)  # 1 for last prediction
        self.out_kb = nn.Linear(self.kb_size, self.kb_size)
        nn.init.xavier_normal_(self.sentinel_gate.weight)
        self.cosine_sim = nn.CosineSimilarity(dim=2)
        self.vocab_size = vocab_size
        self.s_embedding = nn.Embedding(self.vocab_size, emb_dim, padding_idx=0)
        self._cuda = cuda
        #if self.cuda:

        if pretrained_emb is not None:
           self.s_embedding.weight.data.copy_(pretrained_emb)
        self.s_embedding.weight.requires_grad = False
        #self.attention_kb = Attention(hidden_size)
        # Attention
        #self.attention_kb = Attention(emb_dim,emb_dim)

    def forward(self, input_b, input_c, inp_mask,  decoder_out, kb, kb_mask, last_sentient):
        # inp_emb = S X B X E, encoder_hidden = S X 1 X B, decoder_out = S X 1 X B, B X 1
        #                        kb_avg_emb = kb_size X B X E, kb_mask = kb_size X B
        # attention_weights = attention_weights.unsqueeze(1)  # B X S == > B X 1 X S
        # Get average embeddings for the input
        # b_s = attention_weights.size(0)

        encoder_hidden = self.s_embedding(input_b).transpose(0, 1)  # S X B X E
        #encoder_hidden = encoder_hidden.transpose(0, 1)
        decoder_out = self.s_embedding(decoder_out)  # B X E

        input_c = input_c.transpose(0, 1).unsqueeze(1)
        context = input_c.bmm(encoder_hidden)
        context_norm = input_c.sum(2, keepdim=True)
        context = context/(context_norm+1e-10)
        #attn_weights, context = self.attention_kb(encoder_hidden, decoder_out, inp_mask)
        #weighted_inp_emb = .bmm(inp_emb)  # B X 1 X E
        # print (inp_emb_avg.size())
        kb_avg_emb = self.s_embedding(kb).mean(2)  #  B X kb_max_len X EMB
        #print (inp.size())
        # Get cosine similarity with kb subject and relations and multiply with mask
        #kb_cosine = self.cosine_sim(inp_emb_avg, kb_avg_emb)  # B X kb_size
        kb_cosine = self.cosine_sim(context, kb_avg_emb) * kb_mask  # B X kb_size
        #kb_cosine = self.out_kb(F.sigmoid(kb_cosine)) * kb_mask
        inp = torch.cat([(context.squeeze() + decoder_out), kb_cosine, last_sentient.unsqueeze(1)], dim=-1)
        sentient = self.sentinel_gate(inp)  # B X 1
        return sentient, kb_cosine
