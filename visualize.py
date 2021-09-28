import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

with open('logs/models2/4/test_results.txt', 'r') as f:
    lines = f.readlines()

ids = []
generated = []
attentions = []

attention = []
for line in lines:
    if line != '\n':
        if line.split()[0] == 'ID:':
            ids.append(int(line.split()[1]))
        elif line.split()[0] == 'Generated:':
            text = line.split('Generated: ')[1][10:-3]
            text = text.split()
            if text[-1] == '<END>':
                text = text[:-1]
            generated.append(text)
        elif line[:2] == '[[' or line[:2] == '  ' or line[:2] == ' [':
            atts = line.split(line[:2])[-1]
            if ']]' in atts:
                att = atts[:-3].split()
            elif ']' in atts:
                att = atts[:-2].split()
            else:
                att = atts[:-1].split()
            if line[:2] == '[[':
                attention = []
                attention.append([float(i) for i in att])
            elif line[:2] == '  ':
                attention[-1].extend([float(i) for i in att])
            elif line[:2] == ' [':
                attention.append([float(i) for i in att])
            if ']]' in atts:
                attentions.append(attention[1:-1].copy())
df = pd.read_csv('dataset/data2/test.csv')
id = int(input("Visualisasi ID berapa?\n"))
row = df.loc[df['id'] == id]
image_file = row['filename'].item()    
image = Image.open('dataset/data2/images/{}'.format(image_file))

with open('dataset/data2/bounding_box/{}.txt'.format(image_file.split('.')[0]), 'r') as f:
    lines = f.readlines()
bounding_box = []
for line in lines:
    box = line.split()
    x0 = int((float(box[1]) - float(box[3])/2) * image.width)
    x1 = int((float(box[1]) + float(box[3])/2) * image.width)
    y0 = int((float(box[2]) - float(box[4])/2) * image.height)
    y1 = int((float(box[2]) + float(box[4])/2) * image.height)
    bounding_box.append((min(x0,x1), max(y0,y1), abs(x1-x0), -abs(y1-y0)))
    # bounding_box.append((x0, y0, x1, y1))

text = generated[ids.index(id)]
attention = attentions[ids.index(id)]

plt.figure(figsize=(19,10))
for i in range(len(text)):
    plt.subplot(int(np.ceil(len(text)/6)), 6, i+1)
    plt.text(0, 1, text[i], color='black', backgroundcolor='white', fontsize=12)
    plt.imshow(image)
    idx = attention[i].index(max(attention[i]))
    plt.gca().add_patch(Rectangle((bounding_box[idx][0],bounding_box[idx][1]), bounding_box[idx][2], bounding_box[idx][3], linewidth=2, edgecolor='r', facecolor='none'))
    plt.axis('off')

plt.tight_layout()
plt.show()
