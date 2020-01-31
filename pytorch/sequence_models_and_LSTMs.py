#!/usr/bin/env python
# coding: utf-8

# ## Sequence Models and LSTM Networks
# [Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py)

# In[1]:


# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# In[2]:


torch_random_seed = torch.manual_seed(1)


# In[3]:


lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5


# In[4]:


inputs


# In[5]:


# initialize the hidden state.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)


# In[6]:


# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)


# In[7]:


out


# In[8]:


hidden


# In[9]:


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


# In[10]:


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# In[11]:


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()


# In[12]:


# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)


# In[13]:


class LSTMCHARTagger(nn.Module):

    def __init__(self, word_embedding_dim, word_hidden_dim, word_vocab_size, 
                 char_embedding_dim, char_hidden_dim, char_vocab_size, 
                 tagset_size):
        super(LSTMCHARTagger, self).__init__()
        
        # char parameters
        self.char_hidden_dim = char_hidden_dim
        self.char_embedding_dim = char_embedding_dim
        self.char_embeddings = nn.Embedding(char_vocab_size, char_embedding_dim)
        # This LSTM takes char embeddings as inputs, and outputs hidden states
        # with dimensionality char_hidden_dim.
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)        
        
        # word parameters
        self.wordchar_hidden_dim = word_hidden_dim + char_hidden_dim
        self.wordchar_embedding_dim = word_embedding_dim + char_embedding_dim
        self.word_embeddings = nn.Embedding(word_vocab_size, word_embedding_dim)
        # This LSTM takes word embeddings concatenated w/ char embeddings
        # as inputs, and outputs hidden states with 
        # dimensionality wordchar_hidden_dim.
        self.wordchar_lstm = nn.LSTM(self.wordchar_embedding_dim, self.wordchar_hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(self.wordchar_hidden_dim, tagset_size)

    def forward(self, word_chars, sentence):
        # char lstm
        char_hidden_states = []
        for word in word_chars:
            char_embeds = self.char_embeddings(word)
            char_lstm_out, _ = self.char_lstm(char_embeds.view(len(word), 1, -1))
            char_hidden_states.append(char_lstm_out[-1])
        char_hidden_states = torch.cat(char_hidden_states).view(len(sentence), -1)
        
        # wordchar lstm
        word_embeds = self.word_embeddings(sentence)
        # concat with char_hidden_states
        wordchar_embeds = torch.cat((char_hidden_states, word_embeds), dim=1)
        wordchar_lstm_out, _ = self.wordchar_lstm(wordchar_embeds.view(len(sentence), 1, -1))
        
        # linear + softmax
        tag_space = self.hidden2tag(wordchar_lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        
        return tag_scores


# In[14]:


char_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        for char in word:
            if char not in char_to_ix:
                char_to_ix[char] = len(char_to_ix)
print(char_to_ix)


# In[15]:


WORD_EMBEDDING_DIM = 6
WORD_HIDDEN_DIM = 6
WORD_VOCAB_SIZE = len(word_to_ix)
CHAR_EMBEDDING_DIM = 5
CHAR_HIDDEN_DIM = 5
CHAR_VOCAB_SIZE = len(char_to_ix) 
TAGSET_SIZE = len(tag_to_ix)


# In[16]:


model = LSTMCHARTagger(WORD_EMBEDDING_DIM, WORD_HIDDEN_DIM, 
                       WORD_VOCAB_SIZE, CHAR_EMBEDDING_DIM, 
                       CHAR_HIDDEN_DIM, CHAR_VOCAB_SIZE,
                       TAGSET_SIZE)


# In[17]:


loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    words = prepare_sequence(training_data[0][0], word_to_ix)
    word_chars = [prepare_sequence(word, char_to_ix) for word in training_data[0][0]]
    tag_scores = model(word_chars, inputs)
    print(tag_scores)


# In[18]:


for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        word_chars_in = [prepare_sequence(word, char_to_ix) for word in sentence]
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(word_chars_in, sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()


# In[19]:


# See what the scores are after training
with torch.no_grad():
    words = prepare_sequence(training_data[0][0], word_to_ix)
    word_chars = [prepare_sequence(word, char_to_ix) for word in training_data[0][0]]
    tag_scores = model(word_chars, inputs)
    
    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)


# In[20]:


result = np.array([[-0.0741, -3.6919, -3.0677],
                    [-4.1307, -0.0222, -5.1364],
                    [-3.3747, -3.9585, -0.0548],
                    [-0.0308, -5.2657, -3.6829],
                    [-5.1192, -0.0154, -4.6729]])


# In[21]:


assert(result.all() == tag_scores.numpy().all())


# In[23]:


type(tag_scores)


# In[24]:


torch.__version__


# In[ ]:




