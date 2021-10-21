import nltk
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class HeritageDataset(Dataset):
    def __init__(self, csv_file, image_dir, bounding_box_dir, vocab, transform, object_transform, train):
        self.dataset = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.bounding_box_dir = bounding_box_dir
        self.vocab = vocab
        self.transform = transform
        self.object_transform = object_transform
        self.train = train
    def __getitem__(self, index):
        id = self.dataset.iloc[index, 0]
        image = Image.open('{}/{}'.format(self.image_dir, self.dataset.iloc[index, 3])).convert('RGB')

        caption = []
        if not self.train:
            caption = [[self.vocab.getIndex('<START>')] for _ in range(5)]
            for i, element in enumerate(caption):
                text = nltk.word_tokenize(self.dataset.iloc[index, 7+i].lower())
                element.extend([self.vocab.getIndex(word) for word in text])
                element.append(self.vocab.getIndex('<END>'))
        else:
            caption = [self.vocab.getIndex('<START>')]
            caption.extend([self.vocab.getIndex(word) for word in nltk.word_tokenize(self.dataset.iloc[index, 7].lower())])
            caption.append(self.vocab.getIndex('<END>'))

        with open('{}/{}'.format(self.bounding_box_dir, self.dataset.iloc[index, 3].replace('jpg', 'txt')), 'r') as f:
            lines = f.readlines()
            objects = []
            class_list = []
            for line in lines:
                box = line.split()
                class_list.append(int(box[0]))
                x0 = int((float(box[1]) - float(box[3])/2) * image.width)
                x1 = int((float(box[1]) + float(box[3])/2) * image.width)
                y0 = int((float(box[2]) - float(box[4])/2) * image.height)
                y1 = int((float(box[2]) + float(box[4])/2) * image.height)
                object = image.crop((x0,y0,x1,y1))
                object = self.object_transform(object)
                objects.append(object)
        objects = torch.stack(objects, 0)
        image = self.transform(image)
        return id, image, caption, objects, class_list
    def __len__(self):
        return self.dataset.shape[0]

def collate_fn(data):
    ids, images, captions, list_objects, class_lists = zip(*data)
    
    ids = list(ids)
    class_lists = list(class_lists)
    images = torch.stack(images, 0)

    lengths = []
    max_len = 0
    padded_captions = None
    if type(captions[0][0]) == list:
        lengths = [[len(element) for element in caption] for caption in captions]
        max_len = np.amax(lengths)
        lengths = torch.tensor(lengths)
        
        padded_captions = torch.zeros(lengths.size(0), 5, max_len).long()
        for i, caption in enumerate(captions):
            for j, element in enumerate(caption):
                length = lengths[i][j]
                padded_captions[i,j,:length] = torch.tensor(element)
    else:
        lengths = [len(caption) for caption in captions]
        max_len = max(lengths)
        lengths = torch.tensor(lengths).unsqueeze(1)

        padded_captions = torch.zeros(lengths.size(0), max_len).long()
        for i, caption in enumerate(captions):
            length = lengths[i][0]
            padded_captions[i,:length] = torch.tensor(caption)
    
    num_objects = [len(objects) for objects in list_objects]
    max_len = max(num_objects)
    num_objects = torch.tensor(num_objects).unsqueeze(1)

    padded_objects = torch.zeros(num_objects.size(0), max_len, list_objects[0].size(1), list_objects[0].size(2), list_objects[0].size(3))
    for i, objects in enumerate(list_objects):
        length = num_objects[i][0]
        padded_objects[i,:length] = objects
    
    return ids, images, padded_captions, lengths, padded_objects, num_objects, class_lists
