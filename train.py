import torch
import time
from model.models import Encoder, DecoderWithAttention, ObjectEncoder
from torch.nn.utils.rnn import pack_padded_sequence

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embed_dim = None
    attention_dim = None
    decoder_dim = None
    vocab_size = None
    object_dim = None
    
    num_filters = None
    conv_kernel = None
    pool_kernel = None
    image_size = None

    epochs = None
    learning_rate = None

    train_loader = None
    val_loader = None

    encoder = Encoder(embed_dim).to(device)
    decoder = DecoderWithAttention(attention_dim, embed_dim, decoder_dim, vocab_size, object_dim).to(device)
    object_encoder = ObjectEncoder(num_filters, conv_kernel, pool_kernel, image_size, object_dim).to(device)

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