import numpy as np
from datasets import load_dataset

data = load_dataset("ylecun/mnist")

def load_data_wrapper():
    tr_d, te_d = data['train'], data['test']
    
    training_data = []
    for training_example in tr_d:
        training_input = np.reshape(training_example['image'], (784,1))
        training_result = vectorized_result(training_example['label'])
        training_data.append((training_input,training_result))

    test_data = []
    for testing_example in te_d:        
        test_input = np.reshape(testing_example['image'], (784,1))
        test_output = testing_example['label']
        test_data.append((test_input, test_output))

    return (training_data, test_data)

def vectorized_result(j):
    e = np.zeros((10,1))
    e[j] = 1
    return e