import pandas as pd
import pickle
import nltk
from dataset.Vocabulary import Vocabulary

df = pd.read_csv('dataset/data3/train.csv')
vocab = Vocabulary()

vocab.addWord('<PAD>')
vocab.addWord('<START>')
vocab.addWord('<END>')
vocab.addWord('<UNK>')

for index, captions in df[['caption1', 'caption2', 'caption3', 'caption4', 'caption5']].iterrows():
    for idx, caption in captions.iteritems():
        caption = nltk.word_tokenize(caption.lower())
        for word in caption:
            vocab.addWord(word)

with open('dataset/data3/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)