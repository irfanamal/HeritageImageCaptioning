import pandas as pd
from PIL import Image

val = [3.144969479602818, 3.600639950734037, 3.0331605354641384, 3.1332762837495616, 3.7815233979926766, 2.998323039558986, 3.1171217355459353, 3.040040131330635, 3.2654155838607646, 3.459574461971818, 2.6000545510471005, 2.7993787590567005, 3.6791569539011726, 3.3064468959643154, 3.054407153389209, 2.894039812042739, 3.3044842898768563, 2.6684779429382135, 3.050899693491533, 3.2604102039388434]
beam_size = sorted(range(1,len(val)+1), reverse=True, key=lambda k: val[k-1])[:10]

cider = [[] for _ in range(63)]
id = []
for size in beam_size:
    with open('logs/models2/4/beam/{}/test_results.txt'.format(size), 'r') as f:
        lines = f.readlines()
    i = 0
    for line in lines:
        if 'CIDER:' in line:
            cider[i].append(float(line.split()[-1]))
            i += 1
        elif 'ID:' in line:
            id.append(int(line.split()[-1]))

beam_id = [[] for _ in range(10)]
for i, scores in enumerate(cider):
    max_score = max(scores)
    for j in range(10):
        if scores[j] == max_score:
            beam_id[j].append(id[i])

for i, beam in enumerate(beam_id):
    print(beam)
    print(beam_size[i])

df = pd.read_csv('dataset/data2/test.csv')

# # By resolution
# sizes = [[] for _ in range(10)]
# for i, beam in enumerate(beam_id):
#     for id in beam:
#         filename = df[df['id'] == id]['filename'].iloc[0]
#         img = Image.open('dataset/data2/images/{}'.format(filename))
#         size = img.height * img.width
#         sizes[i].append(size)

# avg_size = []
# for size in sizes:
#     avg_size.append(sum(size)/len(size))

# import matplotlib.pyplot as plt

# val = zip(beam_size, avg_size)
# val = sorted(val, key = lambda x: x[0])
# beam_size, avg_size = zip(*val)

# plt.plot(beam_size, avg_size)
# plt.show()

# By MOS
import torch
import cv2
import numpy as np
from torchvision import transforms
from iqa.model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

iqa = Vgg16()
iqa.load_model('iqa/FT_live.caffemodel.pt')
iqa.to(device)
iqa.eval()

Num_Patch = 30

mos = [[] for _ in range(10)]
with torch.no_grad():
    for j, beam in enumerate(beam_id):
        for id in beam:    
            filename = df[df['id'] == id]['filename'].iloc[0]
            img = cv2.imread('dataset/data2/images/{}'.format(filename))
            height = img.shape[0]
            width = img.shape[1]
            if height <= width and height < 224:
                img = cv2.resize(img, (int(width*224/height), 224))
            elif width <= height and width < 224:
                img = cv2.resize(img, (224, int(height*224/width)))
            img = np.asarray(img)
            x, y = img.shape[0], img.shape[1]
            patch_list = []
            for i in range(Num_Patch):
                x_p = x-224
                y_p = y-224
                if x_p > 0:
                    x_p = np.random.randint(x_p)
                if y_p > 0:
                    y_p = np.random.randint(y_p)
                patch = img[x_p:(x_p + 224), y_p:(y_p + 224), :]
                patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(dim=0).float().to(device)
                patch_list.append(patch)
            patches = torch.cat(patch_list, dim=0)
            score = iqa(patches)
            pred = torch.mean(score).item()
            mos[j].append(pred)

avg_mos = []
for score in mos:
    avg_mos.append(sum(score)/len(score))

import matplotlib.pyplot as plt

val = zip(beam_size, avg_mos)
val = sorted(val, key = lambda x: x[0])
beam_size, avg_mos = zip(*val)

plt.plot(beam_size, avg_mos)
plt.show()