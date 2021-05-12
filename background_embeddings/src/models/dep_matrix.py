import dynet as dy
import os
import string
import pickle
import numpy as np, signal, traceback


def get_loss(u, sign):
    if sign == 1:
        # From the data
        return -dy.log_sigmoid(u)
    else:
        return -dy.log_sigmoid(-u)


def probability_distribution(count_table, alpha):
    count_all = 0
    for tok in count_table.keys():
        count_all += count_table[tok] ** alpha
    probs = []
    for tok in count_table.keys():
        probs.append((count_table[tok] ** alpha) / count_all)
    return np.array(probs)


#def sample(count, probs):
   # return np.random.choice(len(probs), count, p=probs)


def init_alias(prob):
    """
    Initialise arrays for sampling with the alias method
    :param prob: probability array
    :return: probability table, alias table
    """
    N = prob.size
    # Initialise tables
    U = prob.astype('float64') / prob.sum() * N
    K = np.arange(N)
    # Initialise lists with weight above and below 1
    below = [i for i,x in enumerate(U) if x<1]
    above = [i for i,x in enumerate(U) if x>=1]
    # Fill tables
    # In each iteration, we remove one index from the pair of lists
    while above and below:
        # Take a pair of indices, one above and one below
        i = below.pop()
        j = above.pop()
        # Fill in the tables
        K[i] = j
        # Calculate the remaining weight of j, and put it back in the correct list
        U[j] -= (1 - U[i])
        if U[j] < 1:
            below.append(j)
        else:
            above.append(j)
    # Note the final index will have U=1, up to rounding error
    return U, K

def alias_sample_one(U, K):
    """
    Sample from a categorical distribution, using the alias method
    :param U: probability table
    :param K: alias table
    :return: sample
    """
    # Choose a random index
    i = np.random.randint(U.size)
    # Return the index, or the alias
    if np.random.rand() > U[i]:
        return K[i]
    else:
        return i

def alias_sample(U, K, n=None):
    """
    Sample from a categorical distribution, using the alias method
    :param U: probability table
    :param K: alias table
    :param n: number of samples to draw (int or tuple of ints)
    :return: array of samples
    """
    if n:
        # Choose random indices
        i = np.random.randint(U.size, size=n)
        # Choose whether to return indices or aliases
        switch = (np.random.random(n) > U[i])
        return switch * K[i] + np.invert(switch) * i
    else:
        return alias_sample_one(U, K)



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
vocab_size = pickle.load( open( "vocab_size_down_500.p", "rb" ) )
#print("Vocab size: {}".format(vocab_size))
# Labels size - from file!
labels_size = pickle.load( open( "labels_size.p", "rb" ) )
vocab = pickle.load( open( "vocab_down_500.p", "rb" ) )
count_table = pickle.load( open( "count_table.p", "rb" ) )
dep_collection = pickle.load( open( "dep_collection.p", "rb" ) )

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
    E = m.add_lookup_parameters((vocab_size, dim), init='uniform', scale = 0.5/dim)

    # Output embeddings:
    O = m.add_lookup_parameters((vocab_size, dim), init=0)

    arr = np.zeros((labels_size, dim, dim))
    for i in range(labels_size):
        arr[i] = np.identity(dim)

    # Dependency matrices:
    T = m.add_lookup_parameters((labels_size, dim, dim), init=arr)


# create trainer
trainer = dy.AdagradTrainer(m, 0.001)

max_file = 3400

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
                        """
                        if train_examples % 10000 == 0:
                            with open('losses_dm.txt', 'a') as f:
                                f.write("{}\n".format(loss_val / train_examples))
                        """
        reader.close()    
        if files % 50 == 0:
            m.save("dm_model_300")
    #print(f'Loss at epoch {epoch}: {loss_val}')

