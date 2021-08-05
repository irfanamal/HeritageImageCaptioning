import nltk
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class HeritageDataset(Dataset):
    def __init__(self, csv_file, image_dir, bounding_box_dir, vocab, transform, object_transform):
        self.dataset = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.bounding_box_dir = bounding_box_dir
        self.vocab = vocab
        self.transform = transform
        self.object_transform = object_transform
    def __getitem__(self, index):
        id = self.dataset.iloc[index, 0]
        image = Image.open('{}/{}'.format(self.image_dir, self.dataset.iloc[index, 3])).convert('RGB')

        caption = [self.vocab.getIndex('<START>')]
        caption.extend([self.vocab.getIndex(word) for word in nltk.word_tokenize(self.dataset.iloc[index, 7].lower())])
        caption.append(self.vocab.getIndex('<END>'))
        caption = torch.Tensor(caption)

        with open('{}/{}'.format(self.bounding_box_dir, self.dataset.iloc[index, 3].replace('jpg', 'txt')), 'r') as f:
            lines = f.readlines()
            objects = []
            for line in lines:
                box = line.split()
                x0 = int((float(box[1]) - float(box[3])/2) * image.width)
                x1 = int((float(box[1]) + float(box[3])/2) * image.width)
                y0 = int((float(box[2]) - float(box[4])/2) * image.height)
                y1 = int((float(box[2]) + float(box[4])/2) * image.height)
                object = image.crop((x0,y0,x1,y1))
                object = self.object_transform(object)
                objects.append(object)
        objects = torch.stack(objects, 0)
        image = self.transform(image)
        return id, image, caption, objects
    def __len__(self):
        return self.dataset.shape[0]

def collate_fn(data):
    ids, images, captions, list_objects = zip(*data)

    ids = list(ids)
    images = torch.stack(images, 0)

    lengths = [len(caption) for caption in captions]
    max_len = max(lengths)
    lengths = torch.tensor(lengths).unsqueeze(1)

    padded_captions = torch.zeros(lengths.size(0), max_len).long()
    for i, caption in enumerate(captions):
        length = lengths[i][0]
        padded_captions[i,:length] = caption
    
    num_objects = [len(objects) for objects in list_objects]
    max_len = max(num_objects)
    padded_objects = torch.zeros(len(num_objects), max_len, list_objects[0].size(1), list_objects[0].size(2), list_objects[0].size(3))
    for i, objects in enumerate(list_objects):
        length = num_objects[i]
        padded_objects[i,:length] = objects
    
    return ids, images, padded_captions, lengths, padded_objects
