# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pre import *
from xgbtrain import *
import time
if __name__ == '__main__':
    st = time.time()
    #extract_features('./data/train_add_neg.txt', with_label=1, prefix='./output/')
    print('finish extract features')
    #train('./output/sample-features', './output/sample-features', './output/model')
    print('finish train')
    #extract_features('./data/dsjtzs_txfz_test1.txt', with_label=0, prefix='./output/test')
    print('finish extract features')
    pred = predict('./output/model', './output/testsample-features')
    gen_ans_txt(pred, thresold=0.5, prex = './output/test')
    ed = time.time()
    print(ed - st)