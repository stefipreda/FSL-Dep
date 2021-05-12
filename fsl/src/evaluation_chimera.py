from scipy.stats import spearmanr
import numpy as np


def similarity(word_vec1, word_vec2):
    sim = np.dot(word_vec1, word_vec2) / (np.linalg.norm(word_vec1) * np.linalg.norm(word_vec2))
    return sim


class Evaluation_Chimera:
    def __init__(self, fsl_model, datapath="../datasets/chimera"):
        self.fsl_model = fsl_model
        self.human_scores = {}
        reader = open(datapath + "/ratings.txt", "r", encoding='latin')
        reader.readline()
        for line in reader:
            tokens = line.split()
            hybrid = tokens[5]
            if hybrid not in self.human_scores.keys():
                self.human_scores[hybrid] = []
            sim_word = tokens[7]
            sim_score = tokens[10]
            self.human_scores[hybrid].append((sim_word, sim_score))

        self.context_sentences = {}
        reader = open(datapath + "/dataset_processed.txt", "r", encoding='latin')
        next_chimera = 0
        next_sent_count = 1
        next_sentence = 2
        for i, line in enumerate(reader):
            if i == next_chimera:
                chimera = line.split()[0]
                next_sent_count = i + 1
                next_sentence = i + 2
                sentences = []
            elif i == next_sent_count:
                sent_count = int(line)
                next_chimera = i + sent_count + 2
                next_hybrid = i + sent_count + 1
            elif i == next_hybrid:
                hybrid = line.split()[0]
                if (hybrid, sent_count) not in self.context_sentences.keys():
                    self.context_sentences[(hybrid, sent_count)] = (chimera, sentences)
            elif i == next_sentence:
                sentences.append(line)
                next_sentence = i + 1

    def evaluate(self):
        print("Evaluating Chimera for additive model:")
        for sent_count in [2, 4, 6]:
            scores = []
            for hybrid in self.human_scores.keys():
                human_scores = [score for (word, score) in self.human_scores[hybrid] if
                                word in self.fsl_model.background_model.emb.keys()]
                chimera, contexts = self.context_sentences[(hybrid, sent_count)]
                fsl_vector = self.fsl_model.get_vector(chimera, contexts)

                fsl_scores = [similarity(fsl_vector, self.fsl_model.background_model.emb[word])
                              for (word, score) in self.human_scores[hybrid]
                              if word in self.fsl_model.background_model.emb.keys()
                              if sum(fsl_vector) != 0
                              ]
                if fsl_scores:
                    scores.append(spearmanr(human_scores, fsl_scores)[0])

            print("Spearman for {} context sentences: {}".format(sent_count, np.mean(scores)))
