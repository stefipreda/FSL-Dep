import gensim.downloader
# Download the "glove-twitter-25" embeddings
word2vec = gensim.downloader.load('word2vec-google-news-300')
print("Monkey:")
print(word2vec.most_similar("monkey"))