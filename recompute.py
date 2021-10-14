# with open('logs/models2/4/test_results.txt', 'r') as f:
#     lines = f.readlines()

# bleu1 = []
# bleu2 = []
# bleu3 = []
# bleu4 = []

# for line in lines:
#     if len(line.split()) > 1:
#         head = line.split()[0]
#         value = line.split()[1]
#         if head == 'BLEU-1:':
#             if float(value) > 1:
#                 bleu1.append(1)
#             else:
#                 bleu1.append(float(value))
#         elif head == 'BLEU-2:':
#             if float(value) > 1:
#                 bleu2.append(1)
#             else:
#                 bleu2.append(float(value))
#         elif head == 'BLEU-3:':
#             if float(value) > 1:
#                 bleu3.append(1)
#             else:
#                 bleu3.append(float(value))
#         elif head == 'BLEU-4:':
#             if float(value) > 1:
#                 bleu4.append(1)
#             else:
#                 bleu4.append(float(value))

# with open('logs/models2/4/test_results_recomputed.txt', 'a+') as f:
#     i = 0
#     for line in lines:
#         if len(line.split()) > 1:
#             if line.split()[0] == 'BLEU-1:':
#                 line = 'BLEU-1: {}\n'.format(bleu1[i])
#             elif line.split()[0] == 'BLEU-2:':
#                 line = 'BLEU-2: {}\n'.format(bleu2[i])
#             elif line.split()[0] == 'BLEU-3:':
#                 line = 'BLEU-3: {}\n'.format(bleu3[i])
#             elif line.split()[0] == 'BLEU-4:':
#                 line = 'BLEU-4: {}\n'.format(bleu4[i])
#                 i += 1
#         f.write(line)

# with open('logs/models2/4/test_summary.txt', 'r') as f:
#     lines = f.readlines()

# with open('logs/models2/4/test_summary_recomputed.txt', 'a+') as f:
#     for line in lines:
#         if len(line.split()) > 1:
#             if line.split()[0] == 'BLEU-1:':
#                 line = 'BLEU-1: {}\n'.format(sum(bleu1)/len(bleu1))
#             elif line.split()[0] == 'BLEU-2:':
#                 line = 'BLEU-2: {}\n'.format(sum(bleu2)/len(bleu2))
#             elif line.split()[0] == 'BLEU-3:':
#                 line = 'BLEU-3: {}\n'.format(sum(bleu3)/len(bleu3))
#             elif line.split()[0] == 'BLEU-4:':
#                 line = 'BLEU-4: {}\n'.format(sum(bleu4)/len(bleu4))
#         f.write(line)

from bleu.bleu import Bleu

with open('logs/models2/7/val_results.txt', 'r') as f:
    lines = f.readlines()

res = {}
gts = {}
id = 0
for line in lines:
    if 'ID:' in line:
        id = int(line.split()[-1])
    elif 'Ground Truth:' in line:
        sent = line.split('Ground Truth: ')[-1][2:-3]
        gts[id] = [sent]
    elif 'Generated:' in line:
        sent = line.split('Generated: ')[-1][2:-3]
        res[id] = [sent]
bleu_score = Bleu().compute_score(gts, res)

with open('logs/models2/7/val_results_recomputed.txt', 'a+') as f:
    i = 0
    for line in lines:
        if len(line.split()) > 1:
            if line.split()[0] == 'BLEU-1:':
                line = 'BLEU-1: {}\n'.format(bleu_score[1][0][i])
            elif line.split()[0] == 'BLEU-2:':
                line = 'BLEU-2: {}\n'.format(bleu_score[1][1][i])
            elif line.split()[0] == 'BLEU-3:':
                line = 'BLEU-3: {}\n'.format(bleu_score[1][2][i])
            elif line.split()[0] == 'BLEU-4:':
                line = 'BLEU-4: {}\n'.format(bleu_score[1][3][i])
                i += 1
        f.write(line)

with open('logs/models2/7/val_summary.txt', 'r') as f:
    lines = f.readlines()

with open('logs/models2/7/val_summary_recomputed.txt', 'a+') as f:
    for line in lines:
        if len(line.split()) > 1:
            if line.split()[0] == 'BLEU-1:':
                line = 'BLEU-1: {}\n'.format(bleu_score[0][0])
            elif line.split()[0] == 'BLEU-2:':
                line = 'BLEU-2: {}\n'.format(bleu_score[0][1])
            elif line.split()[0] == 'BLEU-3:':
                line = 'BLEU-3: {}\n'.format(bleu_score[0][2])
            elif line.split()[0] == 'BLEU-4:':
                line = 'BLEU-4: {}\n'.format(bleu_score[0][3])
        f.write(line)