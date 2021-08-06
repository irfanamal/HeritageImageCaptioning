import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

df = pd.read_csv('dataset/dataset.csv')
x = df.values
y = df['id_object'].values

train_val_split = StratifiedShuffleSplit(1, test_size=0.1, random_state=24)
for train_index, val_index in train_val_split.split(x,y):
    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]

val = pd.DataFrame(x_val, columns=['id', 'id_artifact', 'id_object', 'filename', 'article', 'url', 'id_related_image', 'caption'])
val.to_csv('dataset/test.csv', index=False)
train = pd.DataFrame(x_train, columns=['id', 'id_artifact', 'id_object', 'filename', 'article', 'url', 'id_related_image', 'caption'])
train.to_csv('dataset/train.csv', index=False)