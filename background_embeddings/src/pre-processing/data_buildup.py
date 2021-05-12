import pickle
import os
import string


raw_dir = '../../small_training/processed'
parsed_dir = '../../small_training/parsed'
only_dependency = True

raw_files = os.listdir(raw_dir)
parsed_files = os.listdir(parsed_dir)

# Vocabulary vocab {key: word, vocab[key]: position in vocab}
vocab = {}
# Count table, used for negative sampling and removing
count_table = {}
for raw_file in raw_files:
    article = open(raw_dir + '/' + raw_file, 'r', encoding='utf-8').read()
    for line in article.split("\n"):
        tokens = line.split()
        for token in tokens:
            token = token.strip(string.punctuation)
            if token not in count_table.keys() and token not in string.punctuation:
                count_table[token] = 1
            elif token not in string.punctuation:
                count_table[token] += 1
pickle.dump(count_table, open("count_table.p", "wb"))

cnt = 0
for token in count_table.keys():
    freq = count_table[token]
    if freq > 100:
        vocab[token] = cnt
        cnt += 1
vocab_size = cnt
pickle.dump(vocab_size, open("vocab_size.p", "wb"))
pickle.dump(vocab, open("vocab.p", "wb"))


