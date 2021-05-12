import os
import operator

parsed_dir = '../../small_training/parsed'
# parsed_dir = '../big_training/parsed$'

parsed_files = os.listdir(parsed_dir)

words_count = 0
training_tuples = 0
freq = {}

for file in parsed_files:
    reader = open(parsed_dir + '/' + file, 'r', encoding='utf-8')
    for line in reader:
        training_tuples += 1
        words = line.split(" ")
        if len(words) == 3:
            word1 = words[0]
            word2 = words[1]

        if word1 in freq.keys():
            freq[word1] += 1
        else:
            freq[word1] = 1

words_count = len(freq)
print("Words count: {}".format(words_count))
print("Training examples: {}".format(training_tuples))

sorted_freq = sorted(freq.items(), key=operator.itemgetter(1),reverse=True)
for (w, f) in sorted_freq:
    print("{}, freq:{}".format(w, f))