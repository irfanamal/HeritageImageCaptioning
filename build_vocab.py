import pandas as pd
import pickle
import nltk
from dataset.Vocabulary import Vocabulary

df = pd.read_csv('dataset/dataset.csv')
vocab = Vocabulary()

vocab.addWord('<PAD>')
vocab.addWord('<START>')
vocab.addWord('<END>')
vocab.addWord('<UNK>')

for index, caption in df['caption'].iteritems():
    caption = nltk.word_tokenize(caption.lower())
    for word in caption:
        vocab.addWord(word)

with open('dataset/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)