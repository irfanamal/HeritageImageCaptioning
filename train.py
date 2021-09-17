import pickle
import torch
import time
import dataset.HeritageDataset as Dataset
from dataset.Vocabulary import Vocabulary
from model.models2 import GlobalEncoder, ObjectDescriptor, CaptionGenerator
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__=='__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open('dataset/data2/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    embedding_size = 1024
    vocab_size = len(vocab)
    hidden_size = 256

    epochs = 50
    object_size = 64
    experiment_num = 7
    learning_rate = 0.001
    train_batch_size = 8
    val_batch_size = 8

    transform = transforms.Compose([transforms.Resize(224),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform2 = transforms.Compose([transforms.Resize(object_size),transforms.RandomCrop(object_size),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = Dataset.HeritageDataset('dataset/data2/train.csv', 'dataset/data2/images', 'dataset/data2/bounding_box', vocab, transform, transform2)
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, collate_fn=Dataset.collate_fn)
    val_dataset = Dataset.HeritageDataset('dataset/data2/val.csv', 'dataset/data2/images', 'dataset/data2/bounding_box', vocab, transform, transform2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=4, collate_fn=Dataset.collate_fn)

    global_encoder = GlobalEncoder(hidden_size).to(device)
    object_descriptor = ObjectDescriptor().to(device)
    caption_generator = CaptionGenerator(hidden_size, object_descriptor.descriptor_size, vocab_size, embedding_size).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(list(global_encoder.parameters())+list(object_descriptor.parameters())+list(caption_generator.parameters()), lr=learning_rate)

    min_val_loss = float('inf')
    for i in range(epochs):
        global_encoder.train()
        object_descriptor.train()
        caption_generator.train()

        epoch_loss = 0
        count_loss = 0
        start = time.time()
        for ids, images, captions, lengths, objects, num_objects, _ in train_loader:
            objects = objects.to(device)
            num_objects = num_objects.to(device)
            object_proposals, num_objects, sort_ind = object_descriptor(objects, num_objects)

            images = images.to(device)
            images = images[sort_ind]
            h0 = global_encoder(images)

            captions = captions.to(device)
            lengths = lengths.to(device)
            captions = captions[sort_ind]
            lengths = lengths[sort_ind]
            predictions, attentions, captions, lengths, sort_ind = caption_generator(h0, object_proposals, captions, lengths)
            predictions = pack_padded_sequence(predictions, lengths, batch_first=True)[0]
            captions = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            loss = criterion(predictions, captions)
            caption_generator.zero_grad()
            object_descriptor.zero_grad()
            global_encoder.zero_grad()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * captions.size(0)
            count_loss += sum(lengths)

        train_loss = epoch_loss/count_loss
        end = time.time()

        with torch.no_grad():
            caption_generator.eval()
            object_descriptor.eval()
            global_encoder.eval()

            epoch_loss = 0
            count_loss = 0
            for ids, images, captions, lengths, objects, num_objects, _ in val_loader:
                objects = objects.to(device)
                num_objects = num_objects.to(device)
                object_proposals, num_objects, sort_ind = object_descriptor(objects, num_objects)

                images = images.to(device)
                images = images[sort_ind]
                h0 = global_encoder(images)

                captions = captions.to(device)
                lengths = lengths.to(device)
                captions = captions[sort_ind]
                lengths = lengths[sort_ind]
                predictions, attentions, captions, lengths, sort_ind = caption_generator(h0, object_proposals, captions, lengths)
                
                predictions = pack_padded_sequence(predictions, lengths, batch_first=True)[0]
                captions = pack_padded_sequence(captions, lengths, batch_first=True)[0]                

                loss = criterion(predictions, captions)                
                epoch_loss += loss.item() * captions.size(0)
                count_loss += sum(lengths)
            val_loss = epoch_loss/count_loss

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(caption_generator.state_dict(), 'checkpoint/models2/{}/caption_generator.pt'.format(experiment_num))
            torch.save(global_encoder.state_dict(), 'checkpoint/models2/{}/global_encoder.pt'.format(experiment_num))
            torch.save(object_descriptor.state_dict(), 'checkpoint/models2/{}/object_descriptor.pt'.format(experiment_num))

        with open('logs/models2/{}/train_log.txt'.format(experiment_num), 'a+') as f:
            f.write('Epoch {}, Train Loss: {}, Validation Loss: {}, Training Time: {}\n'.format(i+1, train_loss, val_loss, end-start))
        print('Epoch {}, Train Loss: {}, Validation Loss: {}, Training Time: {}'.format(i+1, train_loss, val_loss, end-start))