# IPython log file
import sys
import pre
import xgbtrain as x
    
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
    for i in range(19):
        if i in used:
            continue
        else:
            t = [i for i in used]
            t.append(i)
            pre.extract_features('data/train.txt', with_label=True, prefix="fea_list%s"%(i), feature_used=t)
        
def train(used):
    temp_cache = sys.stdout
    sys.stdout = open('log.txt','w')
    full = [i for i in range(19)]
    sys.stdout = open('log.txt', 'w')
    x.show_cv(prefix='fea_list', lists=list(set(full) - set(used)), num_round=500)
    sys.stdout = temp_cache

def stat(used):
    data = open('log.txt').readlines()
    n = []
    for i in range(19 - len(used)):
        n.append(data[i*502: (i + 1) * 502])
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

def main():
    used = []
    f = open('sel.txt','w')
    while(len(used) < 18):
        generate(used)
        train(used)
        res = stat(used)
        f.write("%s\t%s\t%s\n"%(str(res[0]).split(), res[1], res[2]))
        used.append(int(res[0]))
