# IPython log file
import sys
import pre
import time
import fine_pre
import xgboost as xgb
import xgbtrain as x
from sklearn.datasets import load_svmlight_file

def check(d, file):
    with open(file, 'a') as f:
        f.write('%02d'%(int(d[0])))
        temp = []
        minn = 100
        for line in d[2:]:
            t = line.strip().split('\t')
            test = float(t[1].split(':')[1])
            train = float(t[2].split(':')[1])
            if train < minn:
                minn = train
                mintest = test
        f.write('\t%2.6f\t%2.6f\n'%(mintest, minn))
        return minn, mintest, d[0]
        
def generate(used):
    for i in range(26):
        if i in used:
            continue
        else:
            t = [i for i in used]
            t.append(i)
            pre.extract_features('data/train.txt', with_label=True, prefix="fea_list%s"%(i), feature_used=t)
        
def train(used):
    temp_cache = sys.stdout
    sys.stdout = open('log.txt','w')
    full = [i for i in range(26)]
    sys.stdout = open('log.txt', 'w')
    x.show_cv(prefix='fea_list', lists=list(set(full) - set(used)), num_round=500)
    sys.stdout = temp_cache

def stat(used):
    data = open('log.txt').readlines()
    n = []
    for i in range(26 - len(used)):
        n.append(data[i*302: (i + 1) * 302])
    min_holder = 10
    min_index = -1
    for i in n:
        res = check(i, 'stat%sst.txt'%(len(used) + 1))
        if res[0] < min_holder:
            min_holder = res[0]
            min_test = res[1]
            min_index = res[2]
    with open('stat%sst.txt'%(len(used) + 1), 'a') as f:
        f.write(str(min_index) + ' ' + str(min_holder))
    return min_index, min_test, min_holder

def go(used):
    
    generate(used)
    train(used)
    stat(used)

def train2(used, num, X, y):
    temp_cache = sys.stdout
    sys.stdout = open('log.txt', 'w')
    trial = [i for i in range(num)]
    for i in trial:
        if i in used:
            continue
        else:
            temp = list(set(used) | set([i]))
            dtrain = xgb.DMatrix(X[:, temp], label=y)
            print(i)
            x.show_cv2(dtrain, num_round=300)
    sys.stdout = temp_cache
    
def one_step(used, num, X, y, feature_map):
    with open('sel.txt', 'a') as f:
        train2(used, num, X, y)
        min_index, min_test, min_cv = stat(used)
        min_index = int(min_index.strip())
        f.write('|%s(%2d)|%s|%1.6f|%1.6f|'%(feature_map[min_index], min_index, len(used), min_test, min_cv)) 
        return min_index

def fine_main(num, feature_map):
    used = []
    
    data = load_svmlight_file('fine_sample-features')
    X = data[0].toarray()
    y = data[1]
    with open('sel.txt','w') as f:
        f.write('|Features|num|test-logloss|CV-logloss|\n')
        f.write('|---|---|---|---|')
    while len(used) < num -1:
        print("\n【NOtice】Now is the %s feature \n"%(len(used)))
        st = time.time()
        new_feature = one_step(used, num, X, y, feature_map)
        used.append(new_feature)
        print('Cost %s'%(time.time() - st))

def read_feature_map(prefix='fine_'):
    with open("%sfeature_map"%(prefix)) as f:
        data = f.readlines()
        num = int(data[0].strip())
        feature_map = []
        for i in data[1:]:
            feature_map.append(i.strip())
    return num, feature_map

def gogogo():
    fine_pre.extract_features('./data/train.txt', with_label=True, prefix='fine_')
    num, feature_map = read_feature_map(prefix='fine_')
    fine_main(num, feature_map)

def main():
    used = []
    f = open('sel.txt','w')
    while(len(used) < 25):
        generate(used)
        train(used)
        res = stat(used)
        f.write("%s\t%s\t%s\n"%(str(res[0]).split(), res[1], res[2]))
        used.append(int(res[0]))

if __name__ == '__main__':
    gogogo()
