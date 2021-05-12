import matplotlib.pylab as plt

xs = [50, 100, 200, 300]

fig, axs = plt.subplots(3, 2, figsize=(12,15))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
for ax in axs.flat:
    ax.set(xticks=xs, xlabel="dimensionionality", ylabel="score")

results_men_sg = [0.153, 0.160, 0.168, 0.173]
results_men_dm = [0.16, 0.164, 0.17, 0.179]

results_simlex_sg = [0.231, 0.27, 0.301, 0.308]
results_simlex_dm = []

results_rw_sg = [0, 0, 0, 0]
results_rw_dm = []

results_simverb_sg = [0, 0, 0, 0]
results_simverb_dm = []

results_wsrel_sg = [0, 0, 0, 0]
results_wsrel_dm = []

results_wssim_sg = [0, 0, 0, 0]
results_wssim_dm = []


axs[0, 0].plot(xs, results_men_sg, marker="o")
axs[0, 0].plot(xs, results_men_dm, marker="o")
axs[0, 0].set_title("MEN")

axs[0, 1].plot(xs, results_simlex_sg, marker="o")
axs[0, 1].set_title("SimLex")

axs[1, 0].plot(xs, results_rw_sg, marker="o")
axs[1, 0].set_title("Rare Words")

axs[1, 1].plot(xs, results_simverb_sg, marker="o")
axs[1, 1].set_title("SimVerb")

axs[2, 0].plot(xs, results_wssim_sg, marker="o")
axs[2, 0].set_title("WS Similarity")

axs[2, 1].plot(xs, results_wsrel_sg, marker="o")
axs[2, 1].set_title("WS Relatedness")

fig.savefig("results.png")