import xgbtrain

from random import shuffle
import codecs
def expand_dataset(neg_file='./output/ans0.txt', pos_file='./output/ans1.txt', data_info='./data/dsjtzs_txfz_test1.txt', origin_dataset='./data/train_add_neg'):
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
        sample = data[int(line)][:-1] + ' 0\n'
        new_dataset.append(sample)
    for line in  open(pos_file):
        sample = data[int(line)][:-1] + ' 1\n'
        new_dataset.append(sample)
    for line in open(origin_dataset):
        new_dataset.append(line)
    shuffle(new_dataset)
    with open('./data/new_train.txt','w') as f:
        for line in new_dataset:
            f.write(line)

def add_neg_train(train_file):
    """
    just use the train data (3000) to add the negative samples
    """
    all_list = []
    neg_list = []
    with codecs.open(train_file, 'r', 'utf-8') as f:
        for ln in f.readlines():
            ln = ln.strip()
            ln_list = ln.split()
            all_list.append(ln)
            if int(ln_list[-1]) == 0:
                neg_list.append(ln)


    with codecs.open('./data/train_add_neg', 'w', 'utf-8') as f:
        len_pos = len(all_list) - len(neg_list)
        len_neg = len(neg_list)

        ad = 3000
        for i in range(len_pos / len_neg - 1):
            for ln in neg_list:
                ln  = ln.split()
                ad += 1
                ln[0] = str(ad)
                ln = ' '.join(ln)
                all_list.append(ln)
        shuffle(all_list)
        for ln in all_list:
            f.write(ln + '\n')



if __name__ == '__main__':
    expand_dataset()
    #add_neg_train('./data/train.txt')