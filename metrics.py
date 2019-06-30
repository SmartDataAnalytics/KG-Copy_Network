import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingMetrics():
    def __init__(self, embeddig_dict, itos=None):
        self.embedding_dict = embeddig_dict
        self.itos = itos

    def eval_emb_metrics(self, gold, prediction):
        """
        get the 3 unsupervised metrics
        :param gold:
        :param prediction:
        :return:
        """
        g_emb, g_avg, g_extreme = self.get_metrics(gold)
        p_emb, p_avg, p_extreme = self.get_metrics(prediction)

        #print (g_avg, p_avg)
        embedding_average = cosine_similarity(g_avg.reshape(1, 300), p_avg.reshape(1, 300))
        vector_extrema = cosine_similarity(g_extreme.reshape(1, 300), p_extreme.reshape(1, 300))
        greedy_matching = self.get_greedy_score(g_emb, p_emb)

        return embedding_average, vector_extrema, greedy_matching

    def get_metrics(self, sent):
        """
        Get all the metrics for the sentences
        :param sent:
        :return:
        """
        # get embeddings for all
        embs = [self.get_embedding(word) for word in sent.split()]
        if not embs:
            avg_emb = np.zeros(300)
        #print ([type(a) for a in embs])
        else:
            avg_emb = np.sum(embs, axis=0) / (np.linalg.norm(np.sum(embs, axis=0)) + 1e-12) # average
        #print (embs)
        try:
            maxemb = np.max(embs, axis=0)
            minemb = np.min(embs, axis=0)
            extreme_emb = list(map(lambda x, y: x if ((x>y or x<-y) and y>0) or ((x<y or x>-y) and y<0) else y, maxemb, minemb)) # vector extrema
        except Exception:
            #print (sent)
            #print (embs)
            extreme_emb = np.zeros(300)

        return embs, avg_emb.reshape(-1, 1), np.array(extreme_emb).reshape(-1, 1)

    @staticmethod
    def get_greedy_score(g_emb, pred_emb):
        """
        Get greedy matching score
        :param g_emb:
        :param pred_emb:
        :return:
        """
        try:
            sim_mat = cosine_similarity(g_emb, pred_emb)
            greedy = (sim_mat.max(axis=0).mean() + sim_mat.max(axis=1).mean()) / 2
        except Exception:
            greedy = 0.0


        return greedy

    def get_embedding(self, word):
        """
        return vectors
        :param word:
        :return:
        """
        try:
            return np.array(self.embedding_dict[word]).astype(np.float64)
        except KeyError:
            return np.random.rand(300)

    def get_metrics_batch(self, gold_batch, pred_batch):
        """
        Get results for batches
        :param gold_batch:
        :param pred_batch:
        :return:
        """
        e_a, v_e, g_m = [], [], []
        for g_s, p_s in zip(gold_batch, pred_batch):
            avg, extreme, greedy = self.eval_emb_metrics(g_s, p_s)
            e_a.append(avg)
            v_e.append(extreme)
            g_m.append(greedy)

            #print (g_m)
        try:
            return np.average(e_a), np.average(v_e), np.average(g_m)
        except Exception as e:
            print (np.average(e_a))
            print (np.average(v_e))
            print (np.average(g_m))
            print (e)




