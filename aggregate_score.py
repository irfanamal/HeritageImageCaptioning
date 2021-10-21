import pandas as pd

df = pd.read_csv('dataset/data3/test.csv')

classname = {101: 'Lompat Tali', 102: 'Engklek', 103: 'Engrang', 104: 'Ular Naga', 105: 'Layangan', 106: 'Bakiak', 107: 'Balap Karung', 108: 'Gebuk Bantal', 109: 'Makan Kerupuk', 110: 'Panjat Pinang', 111: 'Tarik Tambang', 112: 'Balap Kelereng', 113: 'Gundu'}

with open('logs/models2/data3/4/test_results.txt', 'r') as f:
    lines = f.readlines()

bleu1 = {}
bleu2 = {}
bleu3 = {}
bleu4 = {}
cider = {}
name = None
for line in lines:
    if 'ID:' in line:
        name = classname[df[df['id']==int(line.split()[-1])]['id_object'].iloc[0]]
    elif 'BLEU-1:' in line:
        if name not in bleu1:
            bleu1[name] = [float(line.split()[-1])]
        else:
            bleu1[name].append(float(line.split()[-1]))
    elif 'BLEU-2:' in line:
        if name not in bleu2:
            bleu2[name] = [float(line.split()[-1])]
        else:
            bleu2[name].append(float(line.split()[-1]))
    elif 'BLEU-3:' in line:
        if name not in bleu3:
            bleu3[name] = [float(line.split()[-1])]
        else:
            bleu3[name].append(float(line.split()[-1]))
    elif 'BLEU-4:' in line:
        if name not in bleu4:
            bleu4[name] = [float(line.split()[-1])]
        else:
            bleu4[name].append(float(line.split()[-1]))
    elif 'CIDER:' in line:
        if name not in cider:
            cider[name] = [float(line.split()[-1])]
        else:
            cider[name].append(float(line.split()[-1]))

with open('logs/models2/data3/4/test_summary_aggregated.txt', 'a+') as f:
    for name in bleu1:
        f.write('{}\n'.format(name))
        f.write('BLEU-1: {}\n'.format(sum(bleu1[name])/len(bleu1[name])))
        f.write('BLEU-2: {}\n'.format(sum(bleu2[name])/len(bleu2[name])))
        f.write('BLEU-3: {}\n'.format(sum(bleu3[name])/len(bleu3[name])))
        f.write('BLEU-4: {}\n'.format(sum(bleu4[name])/len(bleu4[name])))
        f.write('CIDER: {}\n\n'.format(sum(cider[name])/len(cider[name])))