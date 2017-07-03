import xgbtrain

from random import shuffle

def expand_dataset(neg_file='output/ans0.txt', pos_file='output/ans1.txt', data_info='data/dsjtzs_txfz_test1.txt', origin_dataset='data/train.txt'):
    """
    Expand the test set to training set.
    Neg_file stores the ID of negative samples, while pos_file sotres the positives.
    Data_info is the path of training set.
    Will create a new file named new_train.txt which stored the expanded training dataset.
    """
    with open(data_info) as f:
        data = []
        data.append('')
        data.extend(f.readlines())
        new_dataset = []
    for line in  open(neg_file):
        sample = data[int(line)][:-1] + '0\n'
        new_dataset.append(sample)
    for line in  open(pos_file):
        sample = data[int(line)][:-1] + '1\n'
        new_dataset.append(sample)
    for line in open(origin_dataset):
        new_dataset.append(line)
    shuffle(new_dataset)
    with open('data/new_train.txt','w') as f:
        for line in new_dataset:
            f.write(line)
            
    