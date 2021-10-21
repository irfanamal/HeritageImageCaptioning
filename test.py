import nltk
import numpy
import pickle
import torch
import time
import dataset.HeritageDataset as Dataset
from cider.cider import Cider
from bleu.bleu import Bleu
from dataset.Vocabulary import Vocabulary
from model.models2 import GlobalEncoder, ObjectDescriptor, CaptionGenerator
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('dataset/data3/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    embedding_size = 2048
    vocab_size = len(vocab)
    hidden_size = 256

    object_size = 128
    experiment_num = 11
    max_length = 24

    transform = transforms.Compose([transforms.Resize(224),transforms.RandomCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    object_transform = transforms.Compose([transforms.Resize(object_size),transforms.RandomCrop(object_size),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_dataset = Dataset.HeritageDataset('dataset/data3/val.csv', 'dataset/data3/images', 'dataset/data3/bounding_box', vocab, transform, object_transform, False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=Dataset.collate_fn)

    global_encoder = GlobalEncoder(hidden_size).eval().to(device)
    object_descriptor = ObjectDescriptor().eval().to(device)
    caption_generator = CaptionGenerator(hidden_size, object_descriptor.descriptor_size, vocab_size, embedding_size).eval().to(device)

    global_encoder.load_state_dict(torch.load('checkpoint/models2/data3/{}/global_encoder.pt'.format(experiment_num)))
    caption_generator.load_state_dict(torch.load('checkpoint/models2/data3/{}/caption_generator.pt'.format(experiment_num)))
    object_descriptor.load_state_dict(torch.load('checkpoint/models2/data3/{}/object_descriptor.pt'.format(experiment_num)))

    def translate(word_ids):
        text = []
        for id in word_ids:
            if id != 0:
                text.append(vocab.getWord(id))
            if id == 2:
                break
        return text

    results = []
    with torch.no_grad():
        for ids, images, captions, lengths, objects, num_objects, class_lists in test_loader:
            start = time.time()

            objects = objects.to(device)
            num_objects = num_objects.to(device)
            object_proposals, num_objects, sort_ind = object_descriptor(objects, num_objects)

            images = images.to(device)
            h0 = global_encoder(images)

            prediction, attention = caption_generator.predict(h0, object_proposals, max_length)
            generated = translate(prediction.cpu().numpy()[0])
            ground_truth = [translate(captions.numpy()[0][i]) for i in range(5)]
            end = time.time()
            test_time = end-start
            attention = attention[:, :len(generated)].cpu().numpy()[0]
            result = {'id':ids[0], 'generated':generated, 'ground_truth':ground_truth, 'attention':attention, 'test_time':test_time, 'classes':class_lists[0]}
            results.append(result)

    times = []
    gts = {}
    res = {}
    attentions = []
    class_names = []

    with open('dataset/data3/bounding_box/classes.txt', 'r') as f:
        classes = f.readlines()
    for result in results:
        times.append(result['test_time'])
        res[result['id']] = [' '.join(result['generated'])]
        gts[result['id']] = [' '.join(result['ground_truth'][i]) for i in range(5)]
        attentions.append(result['attention'])
        class_names.append([classes[i][:-1] for i in result['classes']])

    cider_score = Cider().compute_score(gts,res)
    bleu_score = Bleu().compute_score(gts,res)
    with open('logs/models2/data3/{}/val_results.txt'.format(experiment_num), 'a+') as f:
        for i,result in enumerate(results):
            f.write('ID: {}\n'.format(result['id']))
            f.write('Ground Truth:\n')
            for j in range(5):
                f.write('{}\n'.format(gts[result['id']][j]))
            f.write('Generated: {}\n'.format(res[result['id']]))
            f.write('Attention:\n{}\n{}\n'.format(class_names[i], attentions[i]))
            f.write('BLEU-1: {}\n'.format(bleu_score[1][0][i]))
            f.write('BLEU-2: {}\n'.format(bleu_score[1][1][i]))
            f.write('BLEU-3: {}\n'.format(bleu_score[1][2][i]))
            f.write('BLEU-4: {}\n'.format(bleu_score[1][3][i]))
            f.write('CIDER: {}\n'.format(cider_score[1][i]))
            f.write('Inference Time: {}\n\n'.format(times[i]))

    with open('logs/models2/data3/{}/val_summary.txt'.format(experiment_num), 'a+') as f:
        f.write('BLEU-1: {}\n'.format(bleu_score[0][0]))
        f.write('BLEU-2: {}\n'.format(bleu_score[0][1]))
        f.write('BLEU-3: {}\n'.format(bleu_score[0][2]))
        f.write('BLEU-4: {}\n'.format(bleu_score[0][3]))
        f.write('CIDER: {}\n'.format(cider_score[0]))
        f.write('Inference Time: {}\n'.format(numpy.mean(times)))

    print('BLEU-1: {}'.format(bleu_score[0][0]))
    print('BLEU-2: {}'.format(bleu_score[0][1]))
    print('BLEU-3: {}'.format(bleu_score[0][2]))
    print('BLEU-4: {}'.format(bleu_score[0][3]))
    print('CIDER: {}'.format(cider_score[0]))
    print('Inference Time: {}'.format(numpy.mean(times)))