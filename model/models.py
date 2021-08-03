# Created by: Sagar Vinodababu
# Modified by: Irfan Ihsanul Amal

import torch
from torch import nn
import torchvision
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, embed_dim):
        super(Encoder, self).__init__()

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Replace linear layer
        resnet.fc = nn.Linear(resnet.fc.in_features, embed_dim)
        self.resnet = resnet

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, embedding_size)
        return out

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, object_dim, decoder_dim, attention_dim):
        """
        :param object_dim: feature size of encoded objects
        :param decoder_dim: size of decoder's LSTM
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(object_dim, attention_dim)  # linear layer to transform encoded object
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, object_out, decoder_hidden):
        """
        Forward propagation.

        :param object_out: encoded objects, a tensor of dimension (batch_size, num_objects, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(object_out)  # (batch_size, num_objects, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_objects)
        alpha = self.softmax(att)  # (batch_size, num_objects)
        attention_weighted_encoding = (object_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

class ObjectEncoder(nn.Module):
    """
    Object Encoder.
    """

    def __init__(self, num_filters, conv_kernel, pool_kernel, image_size, object_dim):
        """
        :param num_filters: number of filters for convolution layer
        :param conv_kernel: kernel size for convolution layer
        :param pool_kernel: kernel size for pooling layer
        :param image_size: size of image input, expected H = w (square image)
        :param object_dim: feature size of encoded object
        """
        super(ObjectEncoder, self).__init__()
        self.conv = nn.Conv2d(3, num_filters, conv_kernel)
        self.pool = nn.MaxPool2d(pool_kernel)
        self.fc = nn.Linear(num_filters * (image_size - conv_kernel + 1)//pool_kernel * (image_size - conv_kernel + 1)//pool_kernel, object_dim)
    
    def forward(self, objects):
        """
        :param objects: object's preprocessed images, a tensor of dimension (batch_size, num_objects, num_channel, height, width)
        :return: encoded object's tensor
        """
        out = torch.zeros(objects.size(0), objects.size(1), self.fc.out_features)  # (batch_size, num_objects, object_dim)
        for i in range(objects.size(1)):
            x = self.conv(objects[:,i])  # (batch_size, num_filters, image_size - conv_kernel + 1, image_size - conv_kernel + 1)
            x = F.relu(x)  # (batch_size, num_filters, image_size - conv_kernel + 1, image_size - conv_kernel + 1)
            x = self.pool(x)  # (batch_size, num_filters, (image_size - conv_kernel + 1)//pool_kernel, (image_size - conv_kernel + 1)//pool_kernel)
            x = torch.flatten(x, 2)  # (batch_size, num_filters * (image_size - conv_kernel + 1)//pool_kernel * (image_size - conv_kernel + 1)//pool_kernel)
            x = self.fc(x)  # (batch_size, object_dim)
            out[:,i] = x
        return out

class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, object_dim, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param object_dim: feature size of encoded objects
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.object_dim = object_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.attention = Attention(object_dim, decoder_dim, attention_dim)  # attention network
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + object_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.f_beta = nn.Linear(decoder_dim, object_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary

    def forward(self, encoder_out, objects, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, embed_dim)
        :param objects: encoded objects, a tensor of dimension (batch_size, num_objects, object_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        num_objects = objects.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        objects = objects[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h = torch.zeros(batch_size, self.decoder_dim)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size, self.decoder_dim)  # (batch_size, decoder_dim)

        decode_lengths = caption_lengths.tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_objects).to(device)

        # At each time-step, decode by
        # attention-weighing objects based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted objects
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(objects[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, object_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            if t == 0:
                h,c = self.decode_step(
                    torch.cat([encoder_out, attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            else:
                h, c = self.decode_step(
                    torch.cat([embeddings[:batch_size_t, t-1, :], attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
