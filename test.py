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
    for z in range(11,21):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open('dataset/data2/vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)

        embedding_size = 512
        vocab_size = len(vocab)
        hidden_size = 256

        object_size = 64
        experiment_num = 4
        max_length = 20
        beam_size = z

        transform = transforms.Compose([transforms.Resize(224),transforms.RandomCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        object_transform = transforms.Compose([transforms.Resize(object_size),transforms.RandomCrop(object_size),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        test_dataset = Dataset.HeritageDataset('dataset/data2/test.csv', 'dataset/data2/images', 'dataset/data2/bounding_box', vocab, transform, object_transform)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=Dataset.collate_fn)

        global_encoder = GlobalEncoder(hidden_size).eval().to(device)
        object_descriptor = ObjectDescriptor().eval().to(device)
        caption_generator = CaptionGenerator(hidden_size, object_descriptor.descriptor_size, vocab_size, embedding_size).eval().to(device)

        global_encoder.load_state_dict(torch.load('checkpoint/models2/{}/global_encoder.pt'.format(experiment_num)))
        caption_generator.load_state_dict(torch.load('checkpoint/models2/{}/caption_generator.pt'.format(experiment_num)))
        object_descriptor.load_state_dict(torch.load('checkpoint/models2/{}/object_descriptor.pt'.format(experiment_num)))

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

                predictions, attentions, probs = caption_generator.beam(h0, object_proposals, max_length, beam_size)
                generated = [translate(predictions.cpu().numpy()[i]) for i in range(beam_size)]
                ground_truth = [translate(captions.numpy()[0])]
                end = time.time()
                test_time = end-start
                attentions = attentions[:, :len(generated[0])].cpu().numpy()[0]
                result = {'id':ids[0], 'generated':generated, 'ground_truth':ground_truth, 'attention':attentions, 'test_time':test_time, 'classes':class_lists[0], 'probs':probs.cpu().numpy()}
                results.append(result)

        times = []
        gts = {}
        res = {}
        attentions = []
        class_names = []

        with open('dataset/data2/bounding_box/classes.txt', 'r') as f:
            classes = f.readlines()
        for result in results:
            times.append(result['test_time'])
            res[result['id']] = [' '.join(result['generated'][0])]
            gts[result['id']] = [' '.join(result['ground_truth'][0])]
            attentions.append(result['attention'])
            class_names.append([classes[i][:-1] for i in result['classes']])

        cider_score = Cider().compute_score(gts,res)
        bleu_score = Bleu().compute_score(gts,res)
        with open('logs/models2/{}/beam/{}/test_results.txt'.format(experiment_num, beam_size), 'a+') as f:
            for i,result in enumerate(results):
                f.write('ID: {}\n'.format(result['id']))
                f.write('Ground Truth: {}\n'.format(gts[result['id']]))
                f.write('Generated:\n')
                for j in range(len(result['generated'])):
                    f.write('{}\n'.format([' '.join(result['generated'][j])]))
                f.write('Probability: {}\n'.format(result['probs']))
                f.write('Attention:\n{}\n{}\n'.format(class_names[i], attentions[i]))
                f.write('BLEU-1: {}\n'.format(bleu_score[1][0][i]))
                f.write('BLEU-2: {}\n'.format(bleu_score[1][1][i]))
                f.write('BLEU-3: {}\n'.format(bleu_score[1][2][i]))
                f.write('BLEU-4: {}\n'.format(bleu_score[1][3][i]))
                f.write('CIDER: {}\n'.format(cider_score[1][i]))
                f.write('Inference Time: {}\n\n'.format(times[i]))

        with open('logs/models2/{}/beam/{}/test_summary.txt'.format(experiment_num, beam_size), 'a+') as f:
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