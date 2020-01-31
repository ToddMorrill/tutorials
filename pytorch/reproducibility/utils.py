import numpy as np
import yaml
import random
import pdb

random.seed(42)
np.random.seed(42)


def generate_rand_data(vocab_size, max_len, num_examples):
    """Generate random data to simulate vocab indices
    
    Args:
        vocab_size (int): number of unique vocabulary words
        max_len (int): length of simulated training examples
        num_examples (int): number of simulated training examples
    
    Yields:
        tuple (np.array, np.array): example, label arrays 
    """
    examples = [[random.randint(1, vocab_size - 1) for _ in range(max_len)]
                for x in range(num_examples)]
    labels = [random.randint(0, 1) for _ in range(num_examples)]
    examples, labels = np.array(examples), np.array(labels)
    for i in range(num_examples):
        yield np.expand_dims(examples[i], 0), np.expand_dims(labels[i], 0)


def load_yaml(filename):
    """Utility to load yaml files
    
    Args:
        filename (str): path to YAML file
    
    Returns:
        dict: dictionary from YAML file
    """
    with open(filename) as infile:
        yaml_dict = yaml.safe_load(infile)
    return yaml_dict
