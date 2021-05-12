datapath="../datasets/chimera"

reader = open(datapath + "/dataset.txt", 'r', encoding='latin')
writer = open(datapath + "/dataset_processed.txt", 'w', encoding='latin')
reader.readline()
odd = True
for line in reader:
    tokens = line.split()
    if odd:
        chimera = tokens[1]
        writer.write(chimera + "\n")
        writing = line.split('.jpg\t')
        sentences = writing[1].split('@@ ')
        writer.write(str(len(sentences)))
        for sentence in sentences:
            writer.write("\n")
            words = sentence.split(" ")
            for i, word in enumerate(words):
                writer.write(word.lower())
                if i < len(words) -1:
                    writer.write(" ")
    if not odd:
        double_concept = tokens[4]
        writer.write(double_concept + '\n')
    odd = not odd
