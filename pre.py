import math
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from pandas import DataFrame, Series


def handle_one(line):
    """
    Input: one line data
    Return: a py-list of 4 items contains [ID, dots series, destination, label]
    if the data is invalid return None
    """
    line = line.strip().split(' ')
    try:
        ID = int(line[0])
        dots = [list(map(eval,i.split(','))) for i in filter(None, line[1].strip().split(';'))]
        dest = [float(i) for i in line[2].strip().split(',')]
        label = int(line[3].strip())
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
    Return: a list of features related to velocity, including [mean, max, min, variance]
    if no velocity can be calculated, return None
    """
    
    v = []
    for i in range(len(dots) - 1): 
        if(dots[i + 1][2] <= dots[i][2]):
            return None
        v.append(float(dist(dots[i + 1], dots[i])) / (dots[i + 1][2] - dots[i][2]))
    if len(v) ==0:
        return None
    v = np.array(v)
    return [v.mean(), v.max(), v.min(), v.var()]

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

def extract_feature(file):
    """
    Extract features 
    Input: dataset filename
    Output: data Id && features  
    v_fs = velocity features
    a_fs = accerlation features
    """
    f = open('sample-features','w')
    for line in open(file):
        sample = handle_one(line)
        ID = sample[0]
        label = sample[3]
        v_fs = get_velocity(sample[1])
        a_fs = get_acc_speed(sample[1])
        if(v_fs == None) or (a_fs == None):
            continue
        f.write('%s\t%s\t%s\n'%(ID, label, ' '.join([str(i) for i in v_fs + a_fs])))