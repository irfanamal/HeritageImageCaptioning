# implementation for areas of attention

import torch
import torchvision

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GLobalEncoder(torch.nn.Module):
    def __init__(self, hidden_size):
        super(GLobalEncoder, self).__init__()
        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        vgg16.classifier[-1] = torch.nn.Linear(vgg16.classifier[-1].in_features, hidden_size)
        self.vgg16 = vgg16
    def forward(self, images):
        return self.vgg16(images)

class ObjectDescriptor(torch.nn.Module):
    def __init__(self):
        super(ObjectDescriptor, self).__init__()
        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        vgg16.features[-1] = torch.nn.AdaptiveMaxPool2d(1)
        self.vgg16 = vgg16.features
    def forward(self, images):
        return self.vgg16(images).squeeze()

class RegionAttention(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, descriptor_size):
        super(RegionAttention, self).__init__()
        self.wh = torch.nn.Linear(embedding_size, hidden_size)
        self.wr = torch.nn.Linear(embedding_size, descriptor_size)
        self.rh = torch.nn.Linear(descriptor_size, hidden_size)
        self.w = torch.nn.Linear(embedding_size, 1)
        self.r = torch.nn.Linear(descriptor_size, 1)
    def forward(self, word_embeddings, object_proposals, hidden_state):
        word_embeddings = word_embeddings.repeat(hidden_state.size(0),1,1)
        hidden_state = hidden_state.unsqueeze(-1)
        wh = self.wh(word_embeddings)
        wh = torch.bmm(wh, hidden_state)
        wr = self.wr(word_embeddings)
        wr = torch.bmm(wr, object_proposals.permute(0,2,1))
        rh = self.rh(object_proposals)
        rh = torch.bmm(rh, hidden_state)
        w = self.w(word_embeddings)
        r = self.r(object_proposals)
        w = torch.add(wh,w)
        r = torch.add(rh,r)
        return w, wr, r

class CaptionGenerator(torch.nn.Module):
    def __init__(self, hidden_size, descriptor_size, vocab_size, embedding_size):
        super(CaptionGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.descriptor_size = descriptor_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.lstm = torch.nn.LSTMCell(embedding_size + descriptor_size, hidden_size)
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.attention = RegionAttention(embedding_size, hidden_size, descriptor_size)
    def forward(self, h0, object_proposals, captions, caption_lengths):
        batch_size = h0.size(0)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        h0 = h0[sort_ind]
        object_proposals = object_proposals[sort_ind]
        captions = captions[sort_ind]

        decode_lengths = caption_lengths.tolist()
        decode_lengths = [i-1 for i in decode_lengths]

        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device)
        attention = torch.zeros(batch_size, max(decode_lengths), object_proposals.size(1)).to(device)
        h = None
        c = None
        region_feedback = None

        embeddings = self.embedding(captions)
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            if t == 0:
                h, c = self.lstm(torch.cat([embeddings[:batch_size_t, t, :], torch.zeros(batch_size_t, self.descriptor_size)], dim=1), (h0[:batch_size_t], h0[:batch_size_t]))
            else:
                h, c = self.lstm(torch.cat([embeddings[:batch_size_t, t, :], region_feedback], dim=1), (h[:batch_size_t], c[:batch_size_t]))
            w, wr, r = self.attention(self.embedding.weight, object_proposals, h)
            preds = torch.add(torch.add(w, torch.sum(wr, 2)), torch.sum(r, 1).unsqueeze(-1))
            predictions[:batch_size_t, t, :] = preds
            region_attention = torch.add(torch.sum(w, 1).unsqueeze(-1), torch.add(torch.sum(wr, 1), r))
            region_attention = torch.nn.functional.softmax(region_attention)
            attention[:batch_size_t, t, :] = region_attention
            region_feedback = torch.bmm(region_attention.unsqueeze(1), object_proposals).squeeze(1)
        return predictions, attention, captions, decode_lengths, sort_ind
    def predict(self, h0, object_proposals, max_length):
        predictions = torch.zeros(1, max_length).long().to(device)
        predictions[:, 0] = torch.LongTensor([1]).to(device)
        attention = torch.zeros(1, max_length, object_proposals.size(1)).to(device)
        h = None
        c = None
        region_feedback = None

        for t in range(max_length):
            embeddings = self.embedding(predictions[:, t])
            if t == 0:
                h, c = self.lstm(torch.cat([embeddings, torch.zeros(1, self.descriptor_size)], dim=1), (h0, h0))
            else:
                h, c = self.lstm(torch.cat([embeddings, region_feedback], dim=1), (h, c))
            w, wr, r = self.attention(self.embedding.weight, object_proposals, h)
            preds = torch.add(torch.add(w, torch.sum(wr, 2)), torch.sum(r, 1).unsqueeze(-1))
            predictions[:, t] = torch.argmax(preds, 1)
            region_attention = torch.add(torch.sum(w, 1).unsqueeze(-1), torch.add(torch.sum(wr, 1), r))
            region_attention = torch.nn.functional.softmax(region_attention)
            attention[:, t, :] = region_attention.squeeze()
            region_feedback = torch.bmm(region_attention.unsqueeze(1), object_proposals).squeeze(1)
        return predictions, attention