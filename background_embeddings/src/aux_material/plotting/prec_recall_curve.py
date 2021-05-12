import pickle
import dynet as dy
import gensim.downloader
from background_embeddings.src.evaluation import Evaluator, sim_rel_curve, load_word_sim_relatedness, cosine
import matplotlib.pyplot as plt

dim = 100
vocab_size = pickle.load(open("../../vocab_size_down_500.p", "rb"))
labels_size = pickle.load(open("../../labels_size.p", "rb"))
vocab = pickle.load(open("../../vocab_down_500.p", "rb"))
freq = pickle.load(open("../freq_3000.p", "rb"))

# Initialize parameters:
m = dy.ParameterCollection()

# Input embeddings:
E = m.add_lookup_parameters((vocab_size, dim))

# Output embeddings:
O = m.add_lookup_parameters((vocab_size, dim))

# Dependency matrices:
T = m.add_lookup_parameters((labels_size, dim, dim))

m.populate("../../dm_model_complete")

evaluator = Evaluator(E, O, vocab, freq, False)
x1, y1 = sim_rel_curve(evaluator)

m.populate("../../dep_sg_model")

evaluator = Evaluator(E, O, vocab, freq, False)
x3, y3 = sim_rel_curve(evaluator)

# Now the pre-trained Word2Vec
word2vec = gensim.downloader.load('word2vec-google-news-300')
embeddings = {}

rel_triples = load_word_sim_relatedness()
sim_triples = load_word_sim_similarity()
scores = []
N = 0
for (w1, w2, score) in sim_triples:
    if (w1, w2, score) not in rel_triples:
        v1 = word2vec[w1]
        v2 = word2vec[w2]
        sim = cosine(v1, v2)
        scores.append((sim, "sim"))
        N += 1
M = 0
for (w1, w2, score) in rel_triples:
    if (w1, w2, score) not in sim_triples:
        v1 = word2vec[w1]
        v2 = word2vec[w2]
        sim = cosine(v1, v2)
        scores.append((sim, "rel"))
        M += 1
scores = sorted(scores, reverse=True)

x2 = []
y2 = []
for K in range(1, N+M+1):
    similar_pairs = 0
    for idx in range(K):
        if scores[idx][1] == "sim":
            similar_pairs += 1
    precision = similar_pairs / K
    recall = similar_pairs / N
    x2.append(recall)
    y2.append(precision)

AUC_DM = round(np.trapz(y1, x1), 3)
AUC_DEP_SG = round(np.trapz(y3, x3), 3)
AUC_SG = round(np.trapz(y2, x2), 3)


plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim(bottom=0, top=1.05)
plt.plot(x1, y1)
plt.plot(x3, y3)
plt.plot(x2, y2)
plt.legend(["DM (AUC = {})".format(AUC_DM), "Dep SG (AUC = {})".format(AUC_DEP_SG), "SG (AUC = {})".format(AUC_SG)])
plt.show()