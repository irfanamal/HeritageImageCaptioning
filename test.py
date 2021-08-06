import nltk
import numpy
import pickle
import torch
import time
import dataset.HeritageDataset as Dataset
from cider.cider import Cider
from dataset.Vocabulary import Vocabulary
from model.models import Encoder, DecoderWithAttention, ObjectEncoder
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('dataset/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    embed_dim = 100
    attention_dim = 100
    decoder_dim = 100
    vocab_size = len(vocab)
    object_dim = 100
    
    num_filters = 3
    conv_kernel = 3
    pool_kernel = 2
    object_size = 32

    max_length = 20

    transform = transforms.Compose([transforms.Resize(224),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    object_transform = transforms.Compose([transforms.Resize(object_size),transforms.RandomCrop(object_size),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_dataset = Dataset.HeritageDataset('dataset/test.csv', 'dataset/images', 'dataset/bounding_box', vocab, transform, object_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=Dataset.collate_fn)

    encoder = Encoder(embed_dim).eval().to(device)
    decoder = DecoderWithAttention(attention_dim, embed_dim, decoder_dim, vocab_size, object_dim).eval().to(device)
    object_encoder = ObjectEncoder(num_filters, conv_kernel, pool_kernel, object_size, object_dim).eval().to(device)

    encoder.load_state_dict(torch.load('checkpoint/encoder.pt'))
    decoder.load_state_dict(torch.load('checkpoint/decoder.pt'))
    object_encoder.load_state_dict(torch.load('checkpoint/object_encoder.pt'))

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
        for ids, images, captions, lengths, objects in test_loader:
            start = time.time()
            images = images.to(device)
            objects = objects.to(device)
            encoded_images = encoder(images)
            encoded_objects = object_encoder(objects)
            predictions, alphas = decoder.predict(encoded_images, encoded_objects, max_length)
            generated = translate(predictions.cpu().numpy()[0])
            ground_truth = [translate(captions.numpy()[0])]
            end = time.time()
            test_time = end-start
            alphas = alphas[:,:len(generated)].cpu().numpy()[0]
            result = {'id':ids[0], 'generated':generated, 'ground_truth':ground_truth, 'alphas':alphas, 'test_time':test_time}
            results.append(result)

    bleu_1s = []
    bleu_2s = []
    bleu_3s = []
    bleu_4s = []
    times = []
    gts = {}
    res = {}
    attentions = []

    for result in results:
        bleu_1s.append(nltk.translate.bleu_score.sentence_bleu(result['ground_truth'], result['generated'], weights=(1,0,0,0), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7))
        bleu_2s.append(nltk.translate.bleu_score.sentence_bleu(result['ground_truth'], result['generated'], weights=(0,1,0,0), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7))
        bleu_3s.append(nltk.translate.bleu_score.sentence_bleu(result['ground_truth'], result['generated'], weights=(0,0,1,0), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7))
        bleu_4s.append(nltk.translate.bleu_score.sentence_bleu(result['ground_truth'], result['generated'], weights=(0,0,0,1), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7))
        times.append(result['test_time'])
        res[result['id']] = [' '.join(result['generated'])]
        gts[result['id']] = [' '.join(result['ground_truth'][0])]
        attentions.append(result['alphas'])

    cider_score = Cider().compute_score(gts,res)
    with open('logs/test_results.txt', 'a+') as f:
        for i,result in enumerate(results):
            f.write('ID: {}\n'.format(result['id']))
            f.write('Ground Truth: {}\n'.format(gts[result['id']]))
            f.write('Generated: {}\n'.format(res[result['id']]))
            f.write('Attention: {}\n'.format(attentions[i]))
            f.write('BLEU-1: {}\n'.format(bleu_1s[i]))
            f.write('BLEU-2: {}\n'.format(bleu_2s[i]))
            f.write('BLEU-3: {}\n'.format(bleu_3s[i]))
            f.write('BLEU-4: {}\n'.format(bleu_4s[i]))
            f.write('CIDER: {}\n'.format(cider_score[1][i]))
            f.write('Inference Time: {}\n\n'.format(times[i]))
    
    with open('logs/test_summary.txt', 'a+') as f:
        f.write('BLEU-1: {}\n'.format(numpy.mean(bleu_1s)))
        f.write('BLEU-2: {}\n'.format(numpy.mean(bleu_2s)))
        f.write('BLEU-3: {}\n'.format(numpy.mean(bleu_3s)))
        f.write('BLEU-4: {}\n'.format(numpy.mean(bleu_4s)))
        f.write('CIDER: {}\n'.format(cider_score[0]))
        f.write('Inference Time: {}\n'.format(numpy.mean(times)))

    print('BLEU-1: {}'.format(numpy.mean(bleu_1s)))
    print('BLEU-2: {}'.format(numpy.mean(bleu_2s)))
    print('BLEU-3: {}'.format(numpy.mean(bleu_3s)))
    print('BLEU-4: {}'.format(numpy.mean(bleu_4s)))
    print('CIDER: {}'.format(cider_score[0]))
    print('Inference Time: {}'.format(numpy.mean(times)))