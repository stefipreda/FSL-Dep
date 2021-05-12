import pickle
import os
import string


raw_dir = '../small_training/processed'
parsed_dir = '../small_training/parsed'
"""
raw_dir = '../big_training/processed'
parsed_dir = '../big_training/parsed'
"""
only_dependency = True

raw_files = os.listdir(raw_dir)
parsed_files = os.listdir(parsed_dir)
if only_dependency:
    # Dependencies collection: {key: dependency, dep_collection[key]: index}
    dep_collection = {}
    cnt = 0

    for parsed_file in parsed_files:
        reader = open(parsed_dir + '/' + parsed_file, 'r', encoding='utf-8')
        for line in reader:
            words = line.split(" ")
            if len(words) == 3:
                dep = words[2]
                if dep != "ROOT\n":  # don't include dependencies between the same token
                    if dep not in dep_collection.keys():
                        dep_collection[dep] = cnt
                        dep_collection[dep + "-1"] = cnt + 1
                        cnt += 2
        pickle.dump(dep_collection, open("dep_collection.p", "wb"))

    labels_size = cnt
    pickle.dump(labels_size, open("labels_size.p", "wb"))
else:
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


