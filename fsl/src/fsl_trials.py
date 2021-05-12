"""
This script tries the Evaluation setup of the tasks DN, CRW, Chimera
on the Additive and Dependency Additive models, with both
pre-trained and Dependency Matrix background models
"""

import pickle
import dynet as dy
import numpy as np
import gensim.downloader
from nltk.corpus import stopwords
from fsl.src.fsl_models import BackgroundModel, AdditiveModel, DependencyAdditiveModel
from fsl.src.evaluation_dn import Evaluation_DN
from fsl.src.evaluation_crw import Evaluation_CRW
from fsl.src.evaluation_chimera import Evaluation_Chimera

vocab_size = pickle.load(open("../../vocab_size_down_500.p", "rb"))
labels_size = pickle.load(open("../../labels_size.p", "rb"))
vocab = pickle.load(open("../../vocab_down_500.p", "rb"))

# Defining the background models:
# Dependency Matrix - load from disk
dim = 100
# Initialize parameters:
m = dy.ParameterCollection()

# Input embeddings:
E = m.add_lookup_parameters((vocab_size, dim))

# Output embeddings:
O = m.add_lookup_parameters((vocab_size, dim))

# Dependency matrices:
T = m.add_lookup_parameters((labels_size, dim, dim))

m.populate("../../dm_model_complete")

embeddings = {}

for word in vocab.keys():
    idx = vocab[word]
    emb = np.concatenate((E[idx].npvalue(), O[idx].npvalue()))
    embeddings[word] = emb

dependency_model = BackgroundModel(embeddings, 2 * dim)

# Pre-trained WOrd2Vec - from Gensim

word2vec = gensim.downloader.load('word2vec-google-news-300')
embeddings = {}
for word in vocab.keys():
    if word in word2vec.vocab:
        emb = word2vec[word]
        embeddings[word] = emb
word2vec_model = BackgroundModel(embeddings, 300)

# Defining the few-shot learning models:

stop_words = set(stopwords.words('english'))

additive_model_dep = AdditiveModel(dependency_model, stop_words)
dependency_additive_model_dep = DependencyAdditiveModel(dependency_model, stop_words)

additive_model_w2v = AdditiveModel(word2vec_model, stop_words)
dependency_additive_model_w2v = DependencyAdditiveModel(word2vec_model, stop_words)

print("Models created.")

# Evaluating each model:
for fsl_model in [additive_model_dep, dependency_additive_model_dep, additive_model_w2v, dependency_additive_model_w2v]:
    evaluator_dn = Evaluation_Chimera(fsl_model)
    evaluator_dn.evaluate()
    evaluator_chimera = Evaluation_Chimera(fsl_model)
    evaluator_chimera.evaluate()
    for count in [1, 2, 4, 8, 16, 32, 64]:
        print("Number of Context Sentences : {}".format(count))
        evaluator = Evaluation_CRW(fsl_model, count)
        evaluator.evaluate()
