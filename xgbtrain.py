import xgboost as xgb
import numpy as np
import operator
import codecs
import time
from collections import defaultdict
from sklearn.datasets import load_svmlight_file

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

def show_cv(prefix='', lists=[], num_round=300):
    param = {'eta': 0.05, 'max_depth': 8, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.5}
    for i in range(len(lists)):
        print(lists[i])
        tune_para(param, "%s%ssample-features"%(prefix, lists[i]), num_round)

def show_cv2(dtrain, num_round=500):
    param = {'eta': 0.05, 'max_depth': 8, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.5}
    print(param)
    res = xgb.cv(param, dtrain, num_boost_round=num_round, nfold=5,
             metrics={'logloss'}, seed = int(time.time()),
             callbacks=[xgb.callback.print_evaluation(show_stdv=False)])
    return res

def tune_para(param, datafile, num_round):
    dtrain = xgb.DMatrix(datafile)

    print(param)
    res = xgb.cv(param, dtrain, num_boost_round=num_round, nfold=5,
             metrics={'logloss'}, seed = int(time.time()),
             callbacks=[xgb.callback.print_evaluation(show_stdv=False)])
    return res
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

def apply_rules(feature_file='./output/testsample-features', feature_map='./output/feature_map'):
    """
    Apply rules to the model outcoming.
    Return two sets, the first one contains the sample judged as positive, the second contains the reverse. 
    """
    pos = set()
    neg = set()
    data = load_svmlight_file(feature_file)
    X = data[0].toarray()
    features = {}
    for idx, line in enumerate(open(feature_map)):
        features[line.strip()] = idx - 1
    pos = pos.union(set(np.where(X[:, features['acc_max_x_only_False_y_only_True']] > 0.25)[0]))    
    pos = pos.union(set(np.where(X[:, features['acc_var_x_only_False_y_only_False']] > 2)[0]))    
    pos = pos.union(set(np.where(X[:, features['h_angle_acc_max']] > 0.02)[0]))    
    pos = pos.union(set(np.where(X[:, features['h_angle_speed_min']] < -0.2)[0]))    
    pos = pos.union(set(np.where(X[:, features['velocity_mean_x_only_False_y_only_True']] > 2.5)[0]))    
    pos = pos.union(set(np.where(X[:, features['go_back_x']] >= 1)[0]))    
    pos = pos.union(set(np.where(X[:, features['acc_min_x_only_False_y_only_False']] < -2)[0]))    
    pos = pos.union(set(np.where(X[:, features['h_angle_acc_var']] > 1)[0]))    
    pos = pos.union(set(np.where(X[:, features['acc_max_x_only_True_y_only_False']] > 0.5)[0]))    
    pos = pos.union(set(np.where(X[:, features['acc_mean_x_only_False_y_only_False']] < -1)[0]))    
    pos = pos.union(set(np.where(X[:, features['h_angle_speed_var']] > 1)[0]))    
    pos = pos.union(set(np.where(X[:, features['acc_max_x_only_False_y_only_False']] > 0.3)[0]))    
    pos = pos.union(set(np.where(X[:, features['smooth_1_0']] > 1.5)[0]))    
    pos = pos.union(set(np.where(X[:, features['acc_mean_x_only_True_y_only_False']] < -1000)[0]))    
    pos = pos.union(set(np.where(X[:, features['velocity_mean_x_only_False_y_only_False']] > 20)[0]))    
    pos = pos.union(set(np.where(X[:, features['acc_min_x_only_False_y_only_True']] < -1)[0]))    
    neg = neg.union(set(np.where(X[:, features['velocity_min_x_only_False_y_only_False']] > 0.5)[0]))
    
    return pos, neg    

def merge_model(pred1, k=0.5):
    score = defaultdict(float)
    for line in open('./data/gbm_prob.txt'):
        i, j = line.strip().split(',')
        i = int(float(i))
        j = float(j)
        score[i] = (1 - j) * (1 - k)
    with open('./output/testid-map') as f:
        idx = np.array([int(i.strip()) for i in f])
    for i, j in enumerate(pred1):
        score[idx[i]] = score[idx[i]] + (1 - pred1[i]) * k
    return score

def gen_ans_txt(pred, thresold=0.8, prex = ''):
    id_map = prex + 'id-map'
    with open(id_map) as f:
        idx = np.array([int(i.strip()) for i in f])
    mask = np.logical_not(pred >= thresold)
    ans = set()
    for i in idx[mask]:
        ans.add(i)
    # pos, neg = apply_rules()
    # print(len(ans & set(idx[list(pos)])))
    # print(len(set(idx[list(neg)]) - ans))
    # ans = ans - set(idx[list(pos)]) 
    # ans = ans.union(set(idx[list(neg)]))
    with open(prex + 'ans.txt', 'w') as f:
        for i in ans:
            f.write("%s\n"%(i))
