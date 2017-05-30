import xgboost as xgb
import numpy as np

def train(traindata, testdata, modelfile):
    param = {'eta': 0.3, 'max_depth': 2, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.5}
    num_round = 100
    dtrain = xgb.DMatrix(traindata)
    dtest = xgb.DMatrix(testdata)
    bst = xgb.train(param, dtrain, num_round, [(dtest, 'eval'),(dtrain, 'train')])
    bst.save_model(modelfile)
    print('Model has been saved as %s\n'%(modelfile))
    return bst


def tune_para(para, datafile):
    dtrain = xgb.DMatrix(datafile)
    num_round = 2
    print(para)
    res = xgb.cv(param, dtrain, num_boost_round=10, nfold=5,
             metrics={'error'}, seed = 0,
             callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                        xgb.callback.early_stop(3)])

def predict(modelfile, datafile):
    bst = xgb.Booster(model_file=modelfile)
    test = xgb.DMatrix(datafile)
    pred = bst.predict(datafile)

def print_eval(pred, labels):
    corr = (pred > 0.5) == labels
    acc = sum(corr) / len(labels)
    pre_neg_sum = len(labels) - sum(labels)
    true_neg_sum = len(pred) - sum(pred > 0.5)
    neg_pos = sum(np.logical_and(corr, np.logical_not(labels)))
    precision = neg_pos / pre_neg_sum
    recall = neg_pos / true_neg_sum
    print('Acc:%s\tPrecision:%s\tRecall:%s'%(acc, precision, recall))
