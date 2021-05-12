import os
import random
import numpy as np
from scipy.stats import spearmanr


def similarity(word_vec1, word_vec2):
    sim = np.dot(word_vec1, word_vec2) / (np.linalg.norm(word_vec1) * np.linalg.norm(word_vec2))
    return sim


class Evaluation_CRW:
    def __init__(self, fsl_model, sentences_cnt):
        self.fsl_model = fsl_model
        self.sentences_cnt = sentences_cnt
        self.data_path = '../datasets/CRW'

        self.true_similarities = {}
        reader = open(self.data_path + "/CRW-562.txt", "r")
        for line in reader:
            (word1, word2, sim) = line.split()
            self.true_similarities[word2] = (word1, sim)

        self.contexts = {}
        file_paths = os.listdir(self.data_path + "/context")
        for file_path in file_paths:
            rare_word = file_path[0:-4]
            reader = open(self.data_path + "/context/" + file_path, "r", encoding='utf-8')
            contexts = []
            for line in reader:
                contexts.append(line)
            sampled_indices = random.sample(range(0, 254), self.sentences_cnt)
            self.contexts[rare_word] = []
            for idx in sampled_indices:
                self.contexts[rare_word].append(contexts[idx])

    def evaluate(self):
        print("Evaluating CRW for {} model with {} background {}".
              format(self.fsl_model.name, self.fsl_model.background_model.name))
        true_sims = []
        new_sims = []
        for rare_word in self.contexts.keys():
            rare_vector = self.fsl_model.get_vector(rare_word, self.contexts[rare_word])
            (pair_word, true_sim) = self.true_similarities[rare_word]
            if pair_word in self.fsl_model.background_model.emb.keys():
                true_sims.append(true_sim)
                pair_word_vector = self.fsl_model.background_model.emb[pair_word]
                new_sim = similarity(rare_vector, pair_word_vector)
                new_sims.append(new_sim)
        score = spearmanr(true_sims, new_sims)[0]
        print(score)
