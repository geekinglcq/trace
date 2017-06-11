# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pre import *
from xgbtrain import *
if __name__ == '__main__':

    #extract_features('./data/train.txt', with_label=1, prefix='./output/')
    #train('./output/sample-features', './output/sample-features', './output/model')
    #extract_features('./data/dsjtzs_txfz_test1.txt', with_label=0, prefix='./output/test')
    pred = predict('./output/model', './output/testsample-features')
    gen_ans_txt(pred, prex = './output/test')