import pickle
import os
import string


raw_dir = '../../small_training/processed'
parsed_dir = '../../small_training/parsed'
only_dependency = True

raw_files = os.listdir(raw_dir)
parsed_files = os.listdir(parsed_dir)

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