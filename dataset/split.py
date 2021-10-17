import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

df = pd.read_csv('dataset/data3/dataset.csv')
x = df.values
y = df['id_object'].values

train_test_split = StratifiedShuffleSplit(1, test_size=0.1, random_state=24)
for train_index, test_index in train_test_split.split(x,y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

test = pd.DataFrame(x_test, columns=['id', 'id_artifact', 'id_object', 'filename', 'article', 'url', 'id_related_image', 'caption1', 'caption2', 'caption3', 'caption4', 'caption5'])
test.to_csv('dataset/data3/test.csv', index=False)

x = x_train.copy()
y = y_train.copy()
train_val_split = StratifiedShuffleSplit(1, test_size=test.shape[0]/x_train.shape[0], random_state=24)
for train_index, val_index in train_val_split.split(x,y):
    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]

val = pd.DataFrame(x_val, columns=['id', 'id_artifact', 'id_object', 'filename', 'article', 'url', 'id_related_image', 'caption1', 'caption2', 'caption3', 'caption4', 'caption5'])
val.to_csv('dataset/data3/val.csv', index=False)
train = pd.DataFrame(x_train, columns=['id', 'id_artifact', 'id_object', 'filename', 'article', 'url', 'id_related_image', 'caption1', 'caption2', 'caption3', 'caption4', 'caption5'])
train.to_csv('dataset/data3/train.csv', index=False)