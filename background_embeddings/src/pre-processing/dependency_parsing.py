import spacy
import os
import pickle

nlp = spacy.load('en_core_web_sm')

raw_dir_path = '../big_training/processed'
parsed_dir_path = '../big_training/parsed$'

article_paths = os.listdir(raw_dir_path)

vocab = pickle.load(open("vocab_down_500.p", "rb"))

for article_path in article_paths:
    print(article_path)
    article = open(raw_dir_path + '/' + article_path, 'r', encoding='utf-8').read()
    # Create file in train_parsed directory
    file_name = article_path + '_parsed.txt'
    parsed_file = open(os.path.join(parsed_dir_path, file_name), 'w', encoding='utf-8')

    split_text = [article[i:i + 800000] for i in range(0, len(article), 800000)]

    for chunk in split_text:
        doc = nlp(chunk)
        # Write the dependency pairs to the file.
        for token in doc:
            if token.text in vocab.keys( ) and token.head.text in vocab.keys() and token.dep_ != 'punct':
                # don't take into account the dependencies with punctuation                
		# don't take into account spaces
                if token.text != " " and token.head.text != " " and "\n" not in token.text and "\n" not in token.head.text:
                    parsed_file.write(token.text + " " + token.head.text + " " + token.dep_ + "\n")
    parsed_file.close()