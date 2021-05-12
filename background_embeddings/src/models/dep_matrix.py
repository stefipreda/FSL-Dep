import dynet as dy
import os
import pickle
import numpy as np
from background_embeddings.src.utils import probability_distribution, init_alias, alias_sample, get_loss

raw_dir = '../big_training/processed'
parsed_dir = '../big_training/parsed$'

raw_files = os.listdir(raw_dir)
parsed_files = os.listdir(parsed_dir)

# Number of negative samples to take for each correct example:
neg_samples = 10
# Embedding dimensionality:
dim = 300
# Epochs
epochs = 1
# Vocabulary size - from file!
vocab_size = pickle.load(open("vocab_size_down_500.p", "rb"))
# Labels size - from file!
labels_size = pickle.load(open("labels_size.p", "rb"))
vocab = pickle.load(open("vocab_down_500.p", "rb"))
count_table = pickle.load(open("count_table.p", "rb"))
dep_collection = pickle.load(open("dep_collection.p", "rb"))

# Cleanup count_table
to_pop = []
for word in count_table.keys():
    if word not in vocab.keys():
        to_pop.append(word)
for word in to_pop:
    count_table.pop(word)

# Probability distribution
probs = probability_distribution(count_table, 0.75)
U, K = init_alias(probs)
print("Data buildup done.")

# Training starts from this page
with open('files_dm_300.txt', 'r') as f:
    start_file = int(f.readline())

dy.renew_cg()

# Initialize parameters:
m = dy.ParameterCollection()

continue_training = True
if continue_training:
    E = m.add_lookup_parameters((vocab_size, dim))
    O = m.add_lookup_parameters((vocab_size, dim))
    T = m.add_lookup_parameters((labels_size, dim, dim))
    m.populate("dm_model_300")
else:
    # Input embeddings:
    E = m.add_lookup_parameters((vocab_size, dim), init='uniform', scale=0.5 / dim)

    # Output embeddings:
    O = m.add_lookup_parameters((vocab_size, dim), init=0)

    arr = np.zeros((labels_size, dim, dim))
    for i in range(labels_size):
        arr[i] = np.identity(dim)

    # Dependency matrices:
    T = m.add_lookup_parameters((labels_size, dim, dim), init=arr)

# create trainer
trainer = dy.AdagradTrainer(m, 0.025)

max_file = 3000

# Train
for epoch in range(epochs):
    loss_val = 0
    files = 0
    train_examples = 0
    # Go through the input
    for parsed_file in parsed_files:
        reader = open(parsed_dir + '/' + parsed_file, 'r', encoding='utf-8')
        print("Another file")
        files += 1
        if files <= start_file:
            continue
        if files > max_file:
            break
        with open('files_dm_300.txt', 'w') as f:
            f.write(str(files))
            f.flush()
        for line in reader:
            words = line.split(" ")
            train_examples += 1
            if len(words) == 3:
                if words[2] != "ROOT\n":  # don't include dependencies between the same token
                    if words[0] in vocab and words[1] in vocab:
                        # Training pair
                        target_idx = vocab[words[0]]
                        context_idx = vocab[words[1]]
                        label_idx = dep_collection[words[2]]

                        dy.renew_cg()
                        neg_samples_list = alias_sample(U, K, neg_samples)

                        e_t = E[target_idx]
                        t_dep = T[label_idx]
                        o_c = O[context_idx]

                        u = dy.dot_product(e_t, t_dep * o_c)

                        loss = get_loss(u, 1)

                        for false_context_idx in neg_samples_list:
                            o_c = O[false_context_idx]
                            u = dy.dot_product(e_t, t_dep * o_c)
                            loss += get_loss(u, 0)

                        loss_val += loss.value()
                        loss.backward()
                        trainer.update()

                        # Second training pair: inverse dependency
                        target_idx = vocab[words[1]]
                        context_idx = vocab[words[0]]
                        label_idx = dep_collection[words[2] + "-1"]

                        dy.renew_cg()
                        neg_samples_list = alias_sample(U, K, neg_samples)

                        e_t = E[target_idx]
                        t_dep = T[label_idx]
                        o_c = O[context_idx]

                        u = dy.dot_product(e_t, t_dep * o_c)

                        loss = get_loss(u, 1)

                        for false_context_idx in neg_samples_list:
                            o_c = O[false_context_idx]
                            u = dy.dot_product(e_t, t_dep * o_c)
                            loss += get_loss(u, 0)
                        loss_val += loss.value()
                        loss.backward()
                        trainer.update()

                        if train_examples % 10000 == 0:
                            with open('losses_dm.txt', 'a') as f:
                                f.write("{}\n".format(loss_val / train_examples))

        reader.close()
        if files % 50 == 0:
            m.save("dm_model_300")
    print(f'Loss at epoch {epoch}: {loss_val}')
