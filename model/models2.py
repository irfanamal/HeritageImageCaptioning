# implementation for areas of attention

import torch
import torchvision

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GlobalEncoder(torch.nn.Module):
    def __init__(self, hidden_size):
        super(GlobalEncoder, self).__init__()
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
        self.descriptor_size = list(self.vgg16.children())[-3].num_features
    def forward(self, images, num_objects):
        batch_size = images.size(0)

        num_objects, sort_ind = num_objects.squeeze(1).sort(dim=0, descending=True)
        images = images[sort_ind]

        object_lengths = num_objects.tolist()
        object_descriptor = torch.zeros(batch_size, max(object_lengths), self.descriptor_size).to(device)

        for t in range(max(object_lengths)):
            batch_size_t = sum([l > t for l in object_lengths])
            descriptor = self.vgg16(images[:batch_size_t, t, :, :, :]).squeeze()
            object_descriptor[:batch_size_t, t, :] = descriptor

        return object_descriptor, num_objects.unsqueeze(1), sort_ind

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
        return w.squeeze(-1), wr, r.squeeze(-1)

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
                h, c = self.lstm(torch.cat([embeddings[:batch_size_t, t, :], torch.zeros(batch_size_t, self.descriptor_size).to(device)], dim=1), (h0[:batch_size_t], h0[:batch_size_t]))
            else:
                h, c = self.lstm(torch.cat([embeddings[:batch_size_t, t, :], region_feedback[:batch_size_t, :]], dim=1), (h[:batch_size_t], c[:batch_size_t]))
            w, wr, r = self.attention(self.embedding.weight, object_proposals[:batch_size_t, :, :], h)
            preds = torch.add(torch.add(w, torch.sum(wr, 2)), torch.sum(r, 1).unsqueeze(-1))
            predictions[:batch_size_t, t, :] = preds
            region_attention = torch.add(torch.sum(w, 1).unsqueeze(-1), torch.add(torch.sum(wr, 1), r))
            region_attention = torch.nn.functional.softmax(region_attention, dim=1)
            attention[:batch_size_t, t, :] = region_attention
            region_feedback = torch.bmm(region_attention.unsqueeze(1), object_proposals[:batch_size_t, :]).squeeze(1)
        return predictions, attention, captions[:, 1:], decode_lengths, sort_ind
    def predict(self, h0, object_proposals, max_length):
        predictions = torch.zeros(1, max_length).long().to(device)
        predictions[:, 0] = torch.LongTensor([1]).to(device)
        attention = torch.zeros(1, max_length, object_proposals.size(1)).to(device)
        h = None
        c = None
        region_feedback = None

        for t in range(max_length-1):
            embeddings = self.embedding(predictions[:, t])
            if t == 0:
                h, c = self.lstm(torch.cat([embeddings, torch.zeros(1, self.descriptor_size).to(device)], dim=1), (h0, h0))
            else:
                h, c = self.lstm(torch.cat([embeddings, region_feedback], dim=1), (h, c))
            w, wr, r = self.attention(self.embedding.weight, object_proposals, h)
            preds = torch.add(torch.add(w, torch.sum(wr, 2)), torch.sum(r, 1).unsqueeze(-1))
            predictions[:, t+1] = torch.argmax(preds, 1)
            region_attention = torch.add(torch.sum(w, 1).unsqueeze(-1), torch.add(torch.sum(wr, 1), r))
            region_attention = torch.nn.functional.softmax(region_attention, dim=1)
            attention[:, t+1, :] = region_attention.squeeze()
            if predictions[:, t+1].item() == 2:
                break
            region_feedback = torch.bmm(region_attention.unsqueeze(1), object_proposals).squeeze(1)
        return predictions, attention
    def beam(self, h0, object_proposals, max_length, k):
        predictions = torch.zeros(k, max_length).long().to(device)
        attention = torch.zeros(k, max_length, object_proposals.size(1)).to(device)
        probs = torch.zeros(k).to(device)
        lengths = torch.zeros(k).long().to(device)
        h = torch.zeros(k, self.hidden_size).to(device)
        c = torch.zeros(k, self.hidden_size).to(device)
        region_feedback = None
      
        predictions[:, :1] = torch.ones(1).long().to(device)
        probs[:1] = torch.ones(1).to(device)
        lengths[:1] = torch.ones(1).long().to(device)
        h[:1] = h0
        c[:1] = h0
        
        for t in range(max_length-1):
            lengths, sort_ind = lengths.sort(dim=0, descending=True)
            predictions = predictions[sort_ind]
            probs = probs[sort_ind]
            attention = attention[sort_ind]
            h = h[sort_ind]
            c = c[sort_ind]

            batch_size_t = sum([l>t for l in lengths.tolist()])
            embeddings = self.embedding(predictions[:batch_size_t, t])
            if t==0:
                h[:1], c[:1] = self.lstm(torch.cat([embeddings, torch.zeros(1, self.descriptor_size).to(device)], dim=1), (h[:1], c[:1]))
            else:
                region_feedback = torch.bmm(attention[:batch_size_t, t].unsqueeze(1), object_proposals.repeat(batch_size_t,1,1)).squeeze(1)
                h[:batch_size_t], c[:batch_size_t] = self.lstm(torch.cat([embeddings, region_feedback], dim=1), (h[:batch_size_t], c[:batch_size_t]))
            w, wr, r = self.attention(self.embedding.weight, object_proposals.repeat(batch_size_t,1,1), h[:batch_size_t])
            preds = torch.add(torch.add(w, torch.sum(wr, 2)), torch.sum(r, 1).unsqueeze(-1))
            preds = torch.nn.functional.softmax(preds, 1)
            top_k, word_ids = torch.topk(preds, k, dim=1)
            region_attention = torch.add(torch.sum(w, 1).unsqueeze(-1), torch.add(torch.sum(wr, 1), r))
            region_attention = torch.nn.functional.softmax(region_attention, dim=1)
            probs_temp = torch.zeros(batch_size_t*k+k-batch_size_t).to(device)
            sent_temp = torch.zeros(batch_size_t*k+k-batch_size_t, max_length).long().to(device)
            att_temp = torch.zeros(batch_size_t*k+k-batch_size_t, max_length, object_proposals.size(1)).to(device)
            h_temp = torch.zeros(batch_size_t*k+k-batch_size_t, self.hidden_size).to(device)
            c_temp = torch.zeros(batch_size_t*k+k-batch_size_t, self.hidden_size).to(device)
            
            for i in range(batch_size_t):
                sent_temp[k*i:k*(i+1)] = predictions[i].repeat(k,1)
                att_temp[k*i:k*(i+1)] = attention[i].repeat(k,1,1)
                h_temp[k*i:k*(i+1)] = h[i].repeat(k,1)
                c_temp[k*i:k*(i+1)] = c[i].repeat(k,1)
                for j in range(k):
                    prob = probs[i] * top_k[i, j]
                    probs_temp[k*i+j] = prob
                    sent_temp[k*i+j, t+1] = word_ids[i,j]
                    att_temp[k*i+j, t+1] = region_attention[i]
            probs_temp[k*batch_size_t:] = probs[batch_size_t:]
            sent_temp[k*batch_size_t:] = predictions[batch_size_t:]
            att_temp[k*batch_size_t:] = attention[batch_size_t:]
            h_temp[k*batch_size_t:] = h[batch_size_t:]
            c_temp[k*batch_size_t:] = c[batch_size_t:]
            probs_temp, sort_ind = probs_temp.sort(dim=0, descending=True)
            sent_temp = sent_temp[sort_ind]
            att_temp = att_temp[sort_ind]
            h_temp = h_temp[sort_ind]
            c_temp = c_temp[sort_ind]
#             print(probs_temp)
#             print(sent_temp)
#             print(att_temp[:k, :3])
            probs = probs_temp[:k]
            predictions = sent_temp[:k]
            attention = att_temp[:k]
            h = h_temp[:k]
            c = c_temp[:k]
#             print(probs)
#             print(predictions)
#             print(attention)
            lengths_temp = torch.zeros(k).long().to(device)
            end = 0
            for i in range(k):
                count = 0
                for j in range(max_length):
                    if predictions[i, j].item() != 0:
                        if predictions[i, j].item() == 2:
                            end += 1
                            break
                        else:
                            count += 1
                    else:
                        break
                lengths_temp[i] = torch.tensor(count).to(device)
            lengths = lengths_temp
            if end == k:
                break
        return predictions, attention, probs