import yaml
import math
import pdb

import torch
import torch.optim as optim
import numpy as np

from model import ZhangCNNLSTMModel
from utils import generate_rand_data, load_yaml


def train(config):
    """Main training routine
    
    Args:
        config (dict): configuration dictionary for training run
    
    Returns:
        np.array: array of loss metrics
    """
    model = ZhangCNNLSTMModel(config)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    num_examples = 10
    num_batches = math.ceil(num_examples / 1)

    average_loss_tracker = []
    for epoch in range(1, config['max_epoch'] + 1):
        total_loss = 0
        report_loss = 0
        batch_count = 0
        # Iterate over training batches
        for examples, labels in generate_rand_data(config['vocab_size'],
                                                   config['max_len'],
                                                   num_examples):
            labels = torch.Tensor(labels)
            predictions = model(examples)
            loss = loss_fn(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            if 'clip_grad' in config:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              config['clip_grad'])
            optimizer.step()
            report_loss += loss.item()
            total_loss += loss.item()
            batch_count += 1
            if batch_count % config['print_every'] == 0:
                avg_loss = report_loss / config['print_every']
                average_loss_tracker.append(avg_loss)
                print('Epoch: {}, Batch: {}/{}, Average loss: {}'.format(
                    epoch, batch_count, num_batches, avg_loss))
                report_loss = 0
    return np.array(average_loss_tracker)


def main(config):
    average_loss_tracker = train(config)
    return average_loss_tracker


if __name__ == '__main__':
    config = load_yaml('./config.yaml')
    average_loss_tracker = main(config)

    filename = 'avg_loss_tracker'
    np.save(filename, average_loss_tracker)
    load_average_loss_tracker = np.load(filename + '.npy')
    print('saved loss metrics to: {}'.format(filename + '.npy'))
