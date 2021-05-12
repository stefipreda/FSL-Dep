from abc import ABC
import numpy as np
import spacy

class BackgroundModel():
    def __init__(self, embeddings, dim, name):
        self.emb = embeddings
        self.dim = dim
        self.name = name

class FSLModel(ABC):
    def __init__(self, background_model, stop_words, name):
        self.background_model = background_model
        self.stop_words = stop_words
        self.name = name

    # contexts must be a vector of sentences!
    def get_vector(self, word, contexts):
        pass


class AdditiveModel(FSLModel):
    def get_vector(self, word, contexts):
        context_vectors = [self.background_model.emb[x] for sentence in contexts for x in sentence.split() if
                           x != word and x != "___" and x in self.background_model.emb.keys() and x not in self.stop_words]
        return np.zeros(self.background_model.dim) + sum(context_vectors)


class DependencySentenceGraph():
    def __init__(self, sentence):
        self.nlp = spacy.load('en_core_web_sm')
        self.sentence = sentence
        self.neighbours = {}
        doc = self.nlp(sentence)
        for token in doc:
            if token.text != " " and token.head.text != " " and "\n" not in token.text and "\n" not in token.head.text \
                    and token.dep_ != 'ROOT' and token.dep_ != 'punct':
                if token.text not in self.neighbours.keys():
                    self.neighbours[token.text] = [token.head.text]
                    # For now we don't take in account the name of the dependency
                else:
                    self.neighbours[token.text].append(token.head.text)
                if token.head.text not in self.neighbours.keys():
                    self.neighbours[token.head.text] = [token.text]
                    # For now we don't take in account the name of the dependency
                else:
                    self.neighbours[token.head.text].append(token.text)

    def get_depths(self, target):
        # BFS style
        depth = {}
        queue = []
        visited = []
        depth[target] = 0
        queue.append(target)
        visited.append(target)
        while queue:
            top = queue.pop(0)
            if top in self.neighbours.keys():
                for neighbour in self.neighbours[top]:
                    if neighbour not in visited:
                        depth[neighbour] = depth[top] + 1
                        visited.append(neighbour)
                        queue.append(neighbour)
        return depth

    def get_coefficients(self, target):
        depths = self.get_depths(target)
        doc = self.nlp(self.sentence)
        coefs = []
        for token in doc:
            if token.text in stop_words:
                coefs.append(0)
            else:
                if token.text in depths.keys():
                    coefs.append(depths[token.text])
                else:
                    coefs.append(0)
        return coefs

    def get_normalised_coefficients(self, target):
        coefs = self.get_coefficients(target)
        if sum(coefs) == 0:
            norm_coefs = [0 for c in coefs]
        else:
            norm_coefs = [c / sum(coefs) for c in coefs]
        return norm_coefs


class DependencyAdditiveModel(FSLModel):
    def get_vector(self, word, contexts):
        context_vectors = []
        for sentence in contexts:
            graph = DependencySentenceGraph(sentence)
            coefs = graph.get_normalised_coefficients(word)
            idx = 0
            for x in sentence.split():
                if x in self.background_model.emb.keys():
                    if x != word and x != "___" and x not in self.stop_words:
                        context_vectors.append(self.background_model.emb[x] * ( 1 + coefs[idx]))
                idx += 1
        return np.zeros(self.background_model.dim) + sum(context_vectors)


class ALaCarteModel(FSLModel):
    def __init__(self):
        self.A = None

    def get_vector(self, word, contexts):
        context_vectors = [self.background_model[x] for sentence in contexts for x in sentence.split() if
                           x != word and x != "___" and x in self.background_model and x not in self.stop_words]
        return self.A.dot(np.zeros(self.background_model.vector_size) + sum(context_vectors))

