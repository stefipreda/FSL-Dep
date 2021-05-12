import matplotlib.pylab as plt
import pickle
import operator
import math

d = pickle.load(open("../../freq_3000.p", "rb"))

sorted_d = sorted(d.items(), key=operator.itemgetter(1),reverse=True)

x, y = zip(*sorted_d) # unpack a list of pairs into two tuples

#x_top = x[:20]
#y_top = y[:20]
x_nr = range(1, 50001)
y = y[:50000]

x_log = [math.log(i, 10) for i in x_nr]
y_log = [math.log(i, 10) for i in y]

plt.rcParams.update({'font.size': 12})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 7))
ax1.plot(x_nr, y, linewidth=3)
ax1.set_xlabel(xlabel='Rank', labelpad=20)
ax1.set_ylabel(ylabel="Frequency", labelpad=20)
ax1.set_yscale('log')

ax2.plot(x_log, y_log, linewidth=3)
ax2.set_xlabel(xlabel='Log(Rank)', labelpad=20)
ax2.set_ylabel(ylabel="Log(Frequency)", labelpad=20)

"""
for rank in range(5):
    label = x[rank]
    plt.annotate(label,  # this is the text
                 (rank + 1, y[rank]),  # this is the point to label
                 xytext=(0, 10),  # distance from text to points (x,y)
                 ha='left')  # horizontal alignment can be left, right or center
"""
plt.show()