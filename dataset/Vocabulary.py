class Vocabulary():
    def __init__(self):
        self.idx = 0
        self.word2idx = {}
        self.idx2word = {}
    def addWord(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    def getWord(self, idx):
        return self.idx2word[idx]
    def getIndex(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx['<UNK>']
    def size(self):
        return self.idx
    def __len__(self):
        return self.size()