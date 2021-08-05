import pickle
import torch
import time
import dataset.HeritageDataset as Dataset
from dataset.Vocabulary import Vocabulary
from model.models import Encoder, DecoderWithAttention, ObjectEncoder
from torch.nn.utils.rnn import pack_padded_sequence
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

    epochs = 1
    learning_rate = 0.1
    train_batch_size = 3

    transform = transforms.Compose([transforms.Resize(224),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    object_transform = transforms.Compose([transforms.Resize(object_size),transforms.RandomCrop(object_size),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = Dataset.HeritageDataset('dataset/dataset.csv', 'dataset/images', 'dataset/bounding_box', vocab, transform, object_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, collate_fn=Dataset.collate_fn)
    val_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, collate_fn=Dataset.collate_fn)

    encoder = Encoder(embed_dim).to(device)
    decoder = DecoderWithAttention(attention_dim, embed_dim, decoder_dim, vocab_size, object_dim).to(device)
    object_encoder = ObjectEncoder(num_filters, conv_kernel, pool_kernel, object_size, object_dim).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(list(decoder.parameters())+list(object_encoder.parameters())+list(encoder.parameters()), lr=learning_rate)

    min_val_loss = float('inf')
    for i in range(epochs):

        encoder.train()
        decoder.train()
        object_encoder.train()
        epoch_loss = 0
        count_loss = 0
        start = time.time()
        for ids, images, captions, lengths, objects in train_loader:
            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths.to(device)
            objects = objects.to(device)
            encoded_images = encoder(images)
            encoded_objects = object_encoder(objects)
            predictions, captions, lengths, alphas, sort_ind = decoder(encoded_images, encoded_objects, captions, lengths)
            predictions = pack_padded_sequence(predictions, lengths, batch_first=True)[0]
            captions = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            loss = criterion(predictions, captions)
            decoder.zero_grad()
            encoder.zero_grad()
            object_encoder.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * captions.size(0)
            count_loss += sum(lengths)
        train_loss = epoch_loss/count_loss
        end = time.time()

        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            object_encoder.eval()
            epoch_loss = 0
            count_loss = 0
            for ids, images, captions, lengths, objects in val_loader:
                images = images.to(device)
                captions = captions.to(device)
                lengths = lengths.to(device)
                objects = objects.to(device)
                encoded_images = encoder(images)
                encoded_objects = object_encoder(objects)
                predictions, captions, lengths, alphas, sort_ind = decoder(encoded_images, encoded_objects, captions, lengths)
                predictions = pack_padded_sequence(predictions, lengths, batch_first=True)[0]
                captions = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                loss = criterion(predictions, captions)
                epoch_loss += loss.item() * captions.size(0)
                count_loss += sum(lengths)
            val_loss = epoch_loss/count_loss

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(decoder.state_dict(), 'checkpoint/decoder.pt')
            torch.save(encoder.state_dict(), 'checkpoint/encoder.pt')
            torch.save(object_encoder.state_dict(), 'checkpoint/object_encoder.pt')

        print('Epoch {}, Train Loss: {}, Validation Loss: {}, Training Time: {}'.format(i+1, train_loss, val_loss, end-start))