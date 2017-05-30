import math
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from pandas import DataFrame, Series
from itertools import chain


def handle_one(line, with_label=True):
    """
    Input: one line data
    Return: a py-list of 4 items contains [ID, dots series, destination, label]
    Args:
        with_label: if True, the data is train data with labels; if False, the data is test data without labels, set label default to 0
    if the data is invalid return None
    """
    line = line.strip().split(' ')
    try:
        ID = int(line[0])
        dots = [list(map(eval,i.split(','))) for i in filter(None, line[1].strip().split(';'))]
        dest = [float(i) for i in line[2].strip().split(',')]
        if with_label:
            label = int(line[3].strip())
        else:
            label = 0
        return (ID, dots, dest, label)
    except IndexError as e:
        print(line,e)
        return None

def dist(a, b, x_only=False):
    """
    Input: two dots a,b , x_only flag to control if we only take x axis as consideration 
    Return: distance of a and b
    """
    if x_only:
        return abs(b[0] - a[0])
    else:
        return math.sqrt(pow(b[0] - a[0], 2) + pow(b[1] - a[1], 2))

def get_velocity(dots, x_only=False):
    """
    Get the features that related to acceleration
    Input: dots series, x_only flag to control if we only take x axis as consideration 
    Return: a list of features related to velocity, including [mean, max, min, variance, z_per]
    z_per denote the percentage of zero in whole velocity list
    if no velocity can be calculated, return None
    """
    
    v = []
    for i in range(len(dots) - 1): 
        if(dots[i + 1][2] <= dots[i][2]):
            return None
        v.append(float(dist(dots[i + 1], dots[i])) / (dots[i + 1][2] - dots[i][2]))
    if len(v) ==0:
        return None
    z_per = float(sum([1 for i in v if i == 0])) / len(v)
    v = np.array(v)
    return [v.mean(), v.var(), z_per]

def get_acc_speed(dots, x_only=False):
    """
    Get the features that related to acceleration
    Input: dots series, x_only flag to control if we only take x axis as consideration 
    Return: a list of features related to velocity, including [mean, max, min, variance]
    if no acc-speed can be calculated, return None
    """
    v = []
    for i in range(len(dots) - 1): 
        if(dots[i + 1][2] <= dots[i][2]):
            return None
        v.append(float(dist(dots[i + 1], dots[i])) / (dots[i + 1][2] - dots[i][2]))
    if len(v) <= 1:
        return None
    acc = []
    for i in range(len(v) - 1):
        acc.append((v[i + 1] - v[i]) / (dots[i + 1][2] - dots[i][2]))
    acc = np.array(acc)
    return [acc.mean(), acc.max(), acc.min(), acc.var()]

def extract_features(file, with_label=True, prefix=''):
    """
    Extract features and save features in LibSVM format
    Input: dataset filename
    Output: data Id && features  
    v_fs = velocity features
    a_fs = accerlation features
    
    """
    f = open(prefix+'sample-features','w')
    f2 = open(prefix+'id-map','w')
    f3 = open(prefix+'inval-id','w')
    for line in enumerate(open(file)):
        sample = handle_one(line, with_label=with_label)
        ID = sample[0]
        label = sample[3]
        v_fs = get_velocity(sample[1])
        a_fs = get_acc_speed(sample[1])
        if(v_fs == None) or (a_fs == None):
            f3.write('%s\n'%(ID))
            continue
        features = ""
        for i,j in enumerate(chain(v_fs, a_fs)):
            features = features + str(i) + ':' + str(j) + ' '
        f.write('%s %s\n'%(label, features))
        f2.write('%s\n'%(ID))