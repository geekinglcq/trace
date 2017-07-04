import xgboost as xgb
import numpy as np
import operator
import codecs
def train(traindata, testdata, modelfile):
    param = {'eta': 0.3, 'max_depth': 6, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.5}
    num_round = 2000
    dtrain = xgb.DMatrix(traindata)
    dtest = xgb.DMatrix(testdata)
    bst = xgb.train(param, dtrain, num_round, [(dtest, 'eval'),(dtrain, 'train')], early_stopping_rounds = 50)
    feature_importance = bst.get_fscore()
    print(len(feature_importance))
    feature_importance = sorted(feature_importance.items(), key=operator.itemgetter(1))
    #convert feature_importance to feature dictionary

    feature_map_dic = {}
    with codecs.open('./output/' + 'feature_map', 'r', 'utf-8') as f_feature_map:
        cnt_line = 0
        ln = f_feature_map.readline()
        for ln in f_feature_map.readlines():
            f_name = ln.strip()
            feature_map_dic[str('f' + str(cnt_line))] = f_name
            cnt_line += 1
    for fi in feature_importance:
        print(str(feature_map_dic[str(fi[0])]) + ' ' + str(fi[1]))
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
    mask = np.logical_not(pred >= thresold)
    with open(prex + 'ans.txt', 'w') as f:
        for i in idx[mask]:
            f.write('%s\n'%(i))