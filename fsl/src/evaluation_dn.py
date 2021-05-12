import numpy as np


def similarity(word_vec1, word_vec2):
    sim = np.dot(word_vec1, word_vec2) / (np.linalg.norm(word_vec1) * np.linalg.norm(word_vec2))
    return sim


class Evaluation_DN:
    def __init__(self, fsl_model, show):
        self.fsl_model = fsl_model
        self.data_path_train = '../datasets/DN/n2v.definitional.dataset.train.txt'
        self.data_path_test = '../datasets/DN/n2v.definitional.dataset.test.txt'
        self.nonces = {}
        self.show_summary = show

        # Process test file
        reader = open(self.data_path_test, "r")
        for i in range(1, 6):
            reader.readline()
        for line in reader:
            words = line.split()
            nonce = words[0]
            sentence =""
            for c in words[1:]:
                if c == "___":
                    c = nonce
                sentence = sentence + c + " "
            self.nonces[nonce] = sentence

    def run_trial_return_ranks(self):
        ranks = []
        for nonce in self.nonces.keys():
            if self.show_summary:
                print(nonce)
            sentence = self.nonces[nonce]
            nonce_vec = self.fsl_model.get_vector(nonce, [sentence])
            if nonce in self.fsl_model.background_model.emb.keys():
                gold_vec = self.fsl_model.background_model.emb[nonce]
                ranks.append(self.get_rank(nonce_vec, gold_vec, nonce))
        return ranks

    def get_rank(self, nonce_vec, gold_vec, nonce):
        dist_to_gold = similarity(nonce_vec, gold_vec)
        count_closer = 0
        for word in self.fsl_model.background_model.emb.keys():
            if word != nonce:
                dist_to_word = similarity(nonce_vec, self.fsl_model.background_model.emb[word])
                if dist_to_word > dist_to_gold:
                    count_closer += 1
        if self.show_summary:
            print("Rank for nonce: {} is {}".format(nonce, 1 + count_closer))
        return 1 + count_closer

    def evaluate(self):
        print("Evaluating DN for {} model with {} background {}".
              format(self.fsl_model.name, self.fsl_model.background_model.name))
        ranks = self.run_trial_return_ranks()
        mrr = sum([1 / r for r in ranks]) / len(ranks)
        median = np.median(ranks)
        print("MRR is ", mrr)
        print("Median Rank is ", median)
