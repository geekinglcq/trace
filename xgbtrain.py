import xgboost as xgb
import numpy as np

def train(traindata, testdata, modelfile):
    param = {'eta': 0.3, 'max_depth': 6, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.5}
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
    """
    Return predictation of datafile using model stored
    """
    bst = xgb.Booster(model_file=modelfile)
    test = xgb.DMatrix(datafile)
    pred = bst.predict(test)
    return pred


def print_eval(pred, labels):
    """
    Print eval 
    Inputs: predictation and labels
    """
    corr = (pred > 0.5) == labels
    acc = sum(corr) / len(labels)
    pre_neg_sum = len(labels) - sum(labels)
    true_neg_sum = len(pred) - sum(pred > 0.5)
    neg_pos = sum(np.logical_and(corr, np.logical_not(labels)))
    precision = neg_pos / pre_neg_sum
    recall = neg_pos / true_neg_sum
    print('Acc:%s\tPrecision:%s\tRecall:%s'%(acc, precision, recall))

def gen_ans_txt(pred, thresold=0.8, prex = ''):
    id_map = prex + 'id-map'
    with open(id_map) as f:
        idx = np.array([int(i.strip()) for i in f])
    mask = np.logical_not(pred >= 0.8)
    with open(prex + 'ans.txt', 'w') as f:
        for i in idx[mask]:
            f.write('%s\n'%(i))
    