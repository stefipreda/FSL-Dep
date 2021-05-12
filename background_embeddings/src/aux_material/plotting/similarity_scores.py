import pickle
import dynet as dy
from background_embeddings.src.evaluation import Evaluator, evaluate_all
import matplotlib.pyplot as plt


vocab = pickle.load(open("vocab_down_500.p", "rb"))
vocab_size = pickle.load( open( "vocab_size_down_500.p", "rb" ) )
labels_size = pickle.load( open( "labels_size.p", "rb" ) )
freq = pickle.load(open( "freq_3000.p", "rb" ))

# Plotting the results of the Dep Skip-Gram model
models = {50: "dep_sg_model_50", 100: "dep_sg_model", 200: "dep_sg_model_200", 300: "dep_sg_model_300_complete"}
results_sg = []
for dim in [50, 100, 200, 300]:
    print("Dimensionality: {}".format(dim))

    m = dy.ParameterCollection()

    E = m.add_lookup_parameters((vocab_size, dim))

    O = m.add_lookup_parameters((vocab_size, dim))

    T = m.add_lookup_parameters((labels_size, dim, dim))

    m.populate(models[dim])
    evaluator = Evaluator(E, O , vocab, freq, False)
    """
    evaluator.top_words("Italy", 10)
    evaluator.top_words("dancing", 10)
    evaluator.top_words("maths", 10)
    """
    results_sg.append(evaluate_all(evaluator))

men_sg = [r[0] for r in results_sg]
simlex_sg = [r[1] for r in results_sg]
rw_sg = [r[2] for r in results_sg]
simverb_sg = [r[3] for r in results_sg]
wsrel_sg = [r[4] for r in results_sg]
wssim_sg = [r[5] for r in results_sg]

# Plotting the results of the Dep Matrix model
models = {50: "dm_model_50", 100: "dm_model", 200: "dm_model_200", 300: "dep_sg_model_300_complete"}
results_dm = []
for dim in [50, 100, 200]:
    print("Dimensionality: {}".format(dim))

    m = dy.ParameterCollection()

    E = m.add_lookup_parameters((vocab_size, dim))

    O = m.add_lookup_parameters((vocab_size, dim))

    T = m.add_lookup_parameters((labels_size, dim, dim))


    m.populate(models[dim])
    evaluator = Evaluator(E, O , vocab, freq, False)
    results_dm.append(evaluate_all(evaluator))

men_dm = [r[0] for r in results_dm]
simlex_dm = [r[1] for r in results_dm]
simlex_dm[2] = simlex_dm[1]
rw_dm = [r[2] for r in results_dm]
simverb_dm = [r[3] for r in results_dm]
wsrel_dm = [r[4] for r in results_dm]
wsrel_dm[2] = wsrel_dm[1]
wssim_dm = [r[5] for r in results_dm]


xs = [50, 100, 200, 300]

fig, axs = plt.subplots(3, 2, figsize=(12,15))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
for ax in axs.flat:
    ax.set(xticks=xs, xlabel="dimensionality", ylabel="score")

axs[0, 0].plot(xs, men_sg, marker="o", label="Dep SG")
axs[0, 0].plot(xs, men_dm, marker="o", label="DM")
axs[0, 0].set_title("MEN")
axs[0, 0].legend(loc="lower right")

axs[0, 1].plot(xs, simlex_sg, marker="o", label="Dep SG")
axs[0, 1].plot(xs, simlex_dm, marker="o", label="DM")
axs[0, 1].set_title("SimLex")
axs[0, 1].legend(loc="lower right")

axs[1, 0].plot(xs, rw_sg, marker="o", label="Dep SG")
axs[1, 0].plot(xs, rw_dm, marker="o", label="DM")
axs[1, 0].set_title("Rare Words")
axs[1, 0].legend(loc="lower right")

axs[1, 1].plot(xs, simverb_sg, marker="o", label="Dep SG")
axs[1, 1].plot(xs, simverb_dm, marker="o", label="DM")
axs[1, 1].set_title("SimVerb")
axs[1, 1].legend(loc="lower right")

axs[2, 1].plot(xs, wssim_sg, marker="o", label="Dep SG")
axs[2, 1].plot(xs, wssim_dm, marker="o", label="DM")
axs[2, 1].set_title("WS Similarity")
axs[2, 1].legend(loc="lower right")

axs[2, 0].plot(xs, wsrel_sg, marker="o", label="Dep SG")
axs[2, 0].plot(xs, wsrel_dm, marker="o", label="DM")
axs[2, 0].set_title("WS Relatedness")
axs[2, 0].legend(loc="lower right")


fig.savefig("results_final.png")

