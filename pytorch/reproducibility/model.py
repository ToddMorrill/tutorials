import torch
from torch import nn
from torch.nn.modules import Module
import torch.nn.functional as F
import random
import numpy as np

from utils import generate_rand_data, load_yaml

import pdb

torch.manual_seed(42)


class ZhangCNNLSTMModel(Module):
    """
        Model of Zhang et al. 2018 for tweet classification. Uses a single 
        convolutional kernel with local max pooling to reduce input size, then 
        feeds resulting convolutions through an RNN, globally pooling the output
        states to use as the input to a linear layer for classification.
    """
    def __init__(self, config):
        super(ZhangCNNLSTMModel, self).__init__()
        self.vocab_size = config['vocab_size']
        self.embed_size = config['embed_size']
        self.conv_hidden_size = config['conv_hidden_size']
        self.rnn_hidden_size = config['rnn_hidden_size']
        self.kernel_compression = config['kernel_compression']
        # Load pretrained embeddings for the vocab, creating the embedding 
        # dictionary if it does not exist
        self.embedding = nn.Embedding(self.vocab_size,
                                      self.embed_size,
                                      padding_idx=config['pad_id'])
        self.conv = nn.Conv1d(1, self.conv_hidden_size,
                              (self.kernel_compression, self.embed_size))
        self.conv_pool = nn.MaxPool1d(self.kernel_compression,
                                      stride=self.kernel_compression)
        self.word_rnn = nn.LSTM(self.conv_hidden_size,
                                self.rnn_hidden_size,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True)
        self.dropout = nn.Dropout(config['dropout'])
        self.linear = nn.Linear(2 * self.rnn_hidden_size, 1)

    def encode(self, inputs):
        """
            Encode a text sequence using convolutions over the text, followed by
             an RNN-based encoder over the compressed representation.
            Args:
                inputs - list(list(int)) - List of vocab-encoded text sequences
                representing a single batch.
        """
        # B = Batch, C = Channels, L = Length, E = Embedding, H = RNN-Hidden, 
        # F = Filters, K = Kernel Compression
        embeds = self.embedding(torch.tensor(inputs)).unsqueeze(
            1)  # (B, C, L, E)
        dropped_embeds = self.dropout(embeds)
        # Convolutions over word embeddings
        pad = nn.ZeroPad2d((0, 0, 2, 1))
        conv_out = F.relu(self.conv(pad(dropped_embeds)).squeeze(3))
        # Max pool to compress representations
        pooled = self.conv_pool(conv_out)  # (B, F, L/K)
        # Change to (B, L, F) --> (B, H)
        rnn_outs, _ = self.word_rnn(pooled.transpose(1, 2))
        # Global max pool over all LSTM outputs, apply linear layer
        pooled_rnn = F.max_pool1d(rnn_outs.transpose(1, 2),
                                  rnn_outs.size(1)).squeeze(2)
        return pooled_rnn

    def forward(self, inputs):
        """
            Encodes the provided sequences using an RNN encoder and generates a 
            score representing how strongly the sequence is associated with the 
            positive class.
            Args:
                inputs - list(list(int)) - List of vocab-encoded text sequences 
                representing a single batch.
        """
        encoded = self.encode(inputs)
        return self.linear(encoded).squeeze(1)

    def predict(self, inputs, score=False):
        """
            Predict the class membership for each instance in the provided 
            inputs
            Args:
                inputs - list(list(int)) - List of vocab-encoded text sequences
        """
        val = torch.sigmoid(self.forward(inputs))
        if score:
            return val
        return torch.round(val)


if __name__ == '__main__':

    config = load_yaml('./config.yaml')
    model = ZhangCNNLSTMModel(config)
    num_examples = 10
    batch_count = 0
    for examples, labels in generate_rand_data(config['vocab_size'],
                                               config['max_len'],
                                               num_examples):
        preds = model(examples)
        batch_count += 1
    assert (batch_count == num_examples)
