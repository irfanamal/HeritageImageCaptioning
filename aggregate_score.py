import pandas as pd

df = pd.read_csv('dataset/data2/test.csv')
print(df['id_object'].value_counts().sort_index().values)
exit()

classname = {41: 'Ulee Balang', 42: 'Bundo Kanduang', 44: 'Aesan Gede', 45: 'Teluk Belanga', 47: 'Melayu Jambi', 48: 'Paksian', 50: 'Tulang Bawang', 51: 'Pangsi', 52: 'Betawi', 53: 'Kebaya Sunda', 55: 'Jawa', 56: 'Pesaan'}

with open('logs/models2/4/test_results_recomputed.txt', 'r') as f:
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

# with open('logs/models2/4/test_summary_aggregated.txt', 'a+') as f:
#     for name in bleu1:
#         f.write('{}\n'.format(name))
#         f.write('BLEU-1: {}\n'.format(sum(bleu1[name])/len(bleu1[name])))
#         f.write('BLEU-2: {}\n'.format(sum(bleu2[name])/len(bleu2[name])))
#         f.write('BLEU-3: {}\n'.format(sum(bleu3[name])/len(bleu3[name])))
#         f.write('BLEU-4: {}\n'.format(sum(bleu4[name])/len(bleu4[name])))
#         f.write('CIDER: {}\n\n'.format(sum(cider[name])/len(cider[name])))