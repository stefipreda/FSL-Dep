import numpy as np
from scipy.stats import spearmanr
import dynet as dy
import pickle
import matplotlib.pylab as plt
import gensim.downloader


# Cosine similarity between 2 word vectors
def cosine(u, v):
    return np.dot(u, v) / np.sqrt(np.dot(u, u) * np.dot(v, v))

class Evaluator:
    def __init__(self, E1, O1, word_to_idx_map, frequencies, only_E):
        self.E = E1
        self.O = O1
        self.only_input = only_E
        self.freq = frequencies
        self.word_to_idx = word_to_idx_map
        self.default = -1


    def similarity(self, word1, word2):
        if word1 not in self.freq.keys() or word2 not in self.freq.keys():
            # print("Not seen this pair: {}, {}".format(word1, word2))
            return self.default
        print("{} - freq : {} + {} - freq : {}".format(word1, self.freq[word1], word2, self.freq[word2]))
        idx1 = self.word_to_idx[word1]
        idx2 = self.word_to_idx[word2]
        if self.only_input:
            word_vec1 = self.E[idx1].npvalue()
            word_vec2 = self.E[idx2].npvalue()
        else:
            word_vec1 = np.concatenate((self.E[idx1].npvalue(), self.O[idx1].npvalue()))
            word_vec2 = np.concatenate((self.E[idx2].npvalue(), self.O[idx2].npvalue()))

        sim = cosine(word_vec1, word_vec2)
        return sim

    def top_words(self, word, top):
        scores = [(w, self.similarity(word, w)) for w in self.word_to_idx.keys()]
        scores.sort(key=lambda x: x[1])
        top = [w for (w, s) in scores[-top:]]
        print(top)


# Spearman rank correlation between a set of similarity scores assigned by humans and the similarity
# computed using the word embeddings
def evaluate(triples, evaluator):
    # Get vector of gold similarities
    gold = [score for (w1, w2, score) in triples]
    # Get vector of actual similarities
    actual = [evaluator.similarity(w1, w2) for (w1, w2, score) in triples]
    print(spearmanr(gold, actual))


def evaluate_all(evaluator):
    men_triples = load_men()
    sim_lex_triples = load_sim_lex()
    rw_triples = load_rw()
    sim_verb_triples = load_sim_verb()
    ws_relatedness_triples = load_word_sim_relatedness()
    ws_similarity_triples = load_word_sim_similarity()
    print("MEN:")
    evaluate(men_triples, evaluator)
    print("Sim_Lex:")
    evaluate(sim_lex_triples, evaluator)
    print("RW:")
    evaluate(rw_triples, evaluator)
    print("Sim_Verb:")
    evaluate(sim_verb_triples, evaluator)
    print("WS_relatedness:")
    evaluate(ws_relatedness_triples, evaluator)
    print("WS_similarity")
    evaluate(ws_similarity_triples, evaluator)


# Load word similarity_datasets

def load_men():
    # Load MEN dataset
    # Return list of triplets (word1, word2, similarity score)
    men_file = '../testing_data/similarity_datasets/MEN/MEN_dataset_natural_form_full'
    reader = open(men_file, 'r')
    list1 = []
    for line in reader:
        word1, word2, score = line.split()
        list1.append((word1, word2, score))
    return list1


def load_sim_lex():
    # Load SimLex-999 dataset
    # Return list of triplets (word1, word2, similarity score)
    sim_lex_file = '../testing_data/similarity_datasets/SimLex-999/SimLex-999.txt'
    reader = open(sim_lex_file, 'r')
    reader.readline()  # first line is a header
    list1 = []
    for line in reader:
        words = line.split()
        list1.append((words[0], words[1], words[3]))
    return list1


def load_rw():
    # Load RW dataset
    # Return list of triplets (word1, word2, similarity score)
    rw_file = '../testing_data/similarity_datasets/rw/rw.txt'
    reader = open(rw_file, 'r')
    list1 = []
    for line in reader:
        words = line.split()
        list1.append((words[0], words[1], words[2]))
    return list1


def load_sim_verb():
    # Load SimVerb-3500 dataset
    # Return list of triplets (word1, word2, similarity score)
    sim_verb_file = '../testing_data/similarity_datasets/D16-1235.Attachment/data/SimVerb-3500.txt'
    reader = open(sim_verb_file, 'r')
    list1 = []
    for line in reader:
        words = line.split()
        list1.append((words[0], words[1], words[3]))
    return list1


def load_word_sim_relatedness():
    # Load f WordSim353 relatedness dataset
    # Return list of triplets (word1, word2, similarity score)
    word_sim_relatedness_file = '../testing_data/similarity_datasets/wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt'
    reader = open(word_sim_relatedness_file, 'r')
    list1 = []
    for line in reader:
        words = line.split()
        list1.append((words[0], words[1], words[2]))
    return list1


def load_word_sim_similarity():
    # Load f WordSim353 similarity dataset
    # Return list of triplets (word1, word2, similarity score)
    word_sim_similarity_file = '../testing_data/similarity_datasets/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt'
    reader = open(word_sim_similarity_file, 'r')
    list1 = []
    for line in reader:
        words = line.split()
        list1.append((words[0], words[1], words[2]))
    return list1


def sim_rel_scores(evaluator):
    rel_triples = load_word_sim_relatedness()
    sim_triples = load_word_sim_similarity()
    scores = []
    N = 0
    for (w1, w2, score) in sim_triples:
        if (w1, w2, score) not in rel_triples:
            sim = evaluator.similarity(w1, w2)
            scores.append((sim, "sim"))
            N += 1
    M = 0
    for (w1, w2, score) in rel_triples:
        if (w1, w2, score) not in sim_triples:
            sim = evaluator.similarity(w1, w2)
            scores.append((sim, "rel"))
            M += 1
    scores = sorted(scores, reverse=True)
    return scores, N, M


def sim_rel_curve(evaluator):
    scores, N, M = sim_rel_scores(evaluator)
    xs = []
    ys = []
    for K in range(1, N+M+1):
        similar_pairs = 0
        for idx in range(K):
            if scores[idx][1] == "sim":
                similar_pairs += 1
        precision = similar_pairs / K
        recall = similar_pairs / N
        xs.append(recall)
        ys.append(precision)
    return xs, ys

