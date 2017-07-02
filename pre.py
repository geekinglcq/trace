import math
import pandas as pd
import numpy as np
import codecs
import matplotlib.pyplot as plt
import collections

from pandas import DataFrame, Series
from itertools import chain
eps = 1e-6
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

def dist(a, b, x_only=False, y_only = False):
    """
    Input: two dots a,b , x_only flag to control if we only take x axis as consideration 
    Return: distance of a and b
    """
    if x_only:
        return abs(b[0] - a[0])
    if y_only:
        return abs(b[1] - a[1])
    else:
        return math.sqrt(pow(b[0] - a[0], 2) + pow(b[1] - a[1], 2))

def get_dis(dots, x_only = False, y_only = False):
    dots_list = []
    for i in range(len(dots)):
        dots_list.append(float(dist(dots[i], [0, 0], x_only=x_only, y_only=y_only)))
    dots_list = np.array(dots_list)

    #add the feture to the dictionary
    feature_dic = {}
    feature_dic['point_coordinate_mean' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = dots_list.mean()
    feature_dic['point_coordinate_var' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = dots_list.var()
    feature_dic['point_coordinate_in' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = dots_list.min()
    feature_dic['point_coordinate_max' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = dots_list.max()
    return feature_dic

def get_velocity(dots, x_only=False, y_only=False):
    """
    Get the features that related to acceleration
    Input: dots series, x_only flag to control if we only take x axis as consideration 
    Return: a list of features related to velocity, including [mean, max, min, variance, z_per]
    z_per -- denote the percentage of zero in whole velocity list
    v_num -- volumn of velocity points
    if no velocity can be calculated, return None
    """
    
    v = []
    for i in range(len(dots) - 1):
        v.append(float(dist(dots[i + 1], dots[i], x_only=x_only, y_only=y_only)) / (eps + dots[i + 1][2] - dots[i][2]))
    if len(v) ==0:
        return {}
    z_per = float(sum([1 for i in v if i == 0])) / len(v)
    v_num = len(v)
    v = np.array(v)

    #add the feture to the dictionary
    feature_dic = {}
    feature_dic['velocity_mean' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = v.mean()
    feature_dic['velocity_var' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = v.var()
    feature_dic['velocity_min' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = v.min()
    feature_dic['velocity_max' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = v.max()
    feature_dic['velocity_z_per' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = z_per
    feature_dic['velocity_v_num' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = v_num

    return feature_dic

def get_acc_speed(dots, x_only=False, y_only = False):
    """
    Get the features that related to acceleration
    Input: dots series, x_only flag to control if we only take x axis as consideration 
    Return: a list of features related to velocity, including [mean, max, min, variance]
    if no acc-speed can be calculated, return None
    """
    v = []
    for i in range(len(dots) - 1):
        v.append(float(dist(dots[i + 1], dots[i], x_only=x_only, y_only=y_only)) / (eps + dots[i + 1][2] - dots[i][2]))
    if len(v) <= 1:
        return {}
    acc = []
    for i in range(len(v) - 1):
        acc.append((v[i + 1] - v[i]) / (eps + dots[i + 1][2] - dots[i][2]))
    acc = np.array(acc)

    # add the feture to the dictionary
    feature_dic = {}
    feature_dic['acc_mean' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = acc.mean()
    feature_dic['acc_max' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = acc.max()
    feature_dic['acc_min' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = acc.min()
    feature_dic['acc_var' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = acc.var()
    return feature_dic

def get_other_features(dots):
    """
    Features that cannot be classify temporarily
    Including:
    1. If or not the dots in x axis go back out. 1-Y 0-N
    """
    go_back = 0
    for i in range(len(dots) - 1):
        if (dots[i + 1][0] < dots[i][0]):
            go_back = 1

    # add the feture to the dictionary
    feature_dic = {}
    feature_dic['is_x_go_back'] = go_back
    return feature_dic

def dotMinus(dot1, dot2):
    """
    dot2 - dot1 (list - list)
    """
    res = []
    for i in range(len(dot2)):
        res.append(dot2[i] - dot1[i])
    return res
def dotProduct(dot1, dot2):
    """
    dot1 . dot2
    """
    res = 0
    for i in range(len(dot2)):
        res += (dot1[i] * dot2[i])
    return res
def toward_dest(dots, dest_point, x_only=False):
    """
    get the features related to the dots and the destination
    1. the distance
    2. the dot product of the two adjacent vector
    """
    result = []
    dist_to_dest = []
    for i in range(len(dots)):
        dist_to_dest.append(dotMinus(dots[i], dest_point))

    tmp = []
    for i in range(len(dist_to_dest)):
        tmp.append(dist(dist_to_dest[i], [0, 0]))
    tmp = np.array(tmp)
    feature_dic = {}
    if len(tmp):
        # add the feture to the dictionary
        feature_dic['toward_dist_mean'] = tmp.mean()
        feature_dic['toward_dist_min'] = tmp.min()
        feature_dic['toward_dist_max'] = tmp.max()
        feature_dic['toward_dist_var'] = tmp.var()

    tmp = []
    for i in range(len(dist_to_dest) - 1):
        tmp.append(dotProduct(dist_to_dest[i], dist_to_dest[i+1]))
    tmp = np.array(tmp)
    if len(tmp):
        feature_dic['toward_dist_dot_product_mean'] = tmp.mean()
        feature_dic['toward_dist_dot_product_min'] = tmp.min()
        feature_dic['toward_dist_dot_product_max'] = tmp.max()
        feature_dic['toward_dist_dot_product_var'] = tmp.var()
    return feature_dic

def get_other_features(dots):
    """
    Features that cannot be classify temporarily
    Including:
    1. If or not the dots in x axis go back out. 1-Y 0-N
    """
    go_back = 0
    for i in range(len(dots) - 1):
        if (dots[i + 1][0] < dots[i][0]):
            go_back = 1
    
    return [go_back]

def get_density(dots,x_only=True):
    '''
    Features related to the density of points about x-ray
    If there is only one point return 0. If not, return density.
    '''
    x = []
    for i in dots:
        x.append(int(i[0]))
    x = np.array(x)
    if (x.max()-x.min())>=1:
        density = float(len(x))/(x.max()-x.min())
    else:
        density = 0
    return [density]

def get_y_min(dots):
    '''
    Features related to y-values.
    Return the minimum of y-values.
    ''' 
    y = []
    for i in dots:
        y.append(int(i[1]))
    y = np.array(y)
    return [y.min()]

def get_angle_change(dots):
    """
    get the angle changes along the path
    """
    point_minus = []
    for i in range(len(dots) - 1):
        point_minus.append(dotMinus(dots[i+1], dots[i]))

    tmp = []
    if len(point_minus) == 1:
        tmp.append(0)
    else:
        for i in range(len(point_minus) - 1):
            tmp.append(dotProduct(point_minus[i], point_minus[i+1]) / (eps +
                                        dist(point_minus[i], [0, 0]) * dist(point_minus[i+1], [0, 0])))
    feature_dic = {}
    tmp = np.array(tmp)
    # add the feture to the dictionary
    if len(tmp):
        feature_dic['angle_mean'] = tmp.mean()
        feature_dic['angle_min'] = tmp.min()
        feature_dic['angle_max'] = tmp.max()
        feature_dic['angle_var'] = tmp.var()
    return feature_dic
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
    with codecs.open(file, 'r', 'utf-8') as fdata:
        for line in fdata.readlines():
            line = line.strip()
            sample = handle_one(line, with_label=with_label)

            #sort the dots according to ascending time order
            #it seems lables of dots in the descending order time is all 1(not rebot), use this part data
            dots = np.array(sample[1])
            dots = sorted(dots, key=lambda x: x[2])
            for i in range(len(dots)):
                dots[i] = list(dots[i])

            ID = sample[0]
            label = sample[3]

            feature_dict = dict()
            #point coordinate features
            point_coordx_f = get_dis(dots, x_only=True)
            feature_dict = dict(point_coordx_f, **feature_dict)

            point_coordy_f = get_dis(dots, y_only=True)
            feature_dict = dict(point_coordy_f, **feature_dict)

            #velocity features
            v_fs = get_velocity(dots) #4
            feature_dict = dict(v_fs, **feature_dict)

            v_fs_x_only = get_velocity(dots, x_only=True)#4
            feature_dict = dict(v_fs_x_only, **feature_dict)

            v_fs_y_only = get_velocity(dots, y_only=True)#4
            feature_dict = dict(v_fs_y_only, **feature_dict)

            #acc features
            a_fs_y_only = get_acc_speed(dots, y_only=True)  # 4
            feature_dict = dict(a_fs_y_only, **feature_dict)

            a_fs_x_only = get_acc_speed(dots, x_only=True)#4
            feature_dict = dict(a_fs_x_only, **feature_dict)

            a_fs = get_acc_speed(dots)#4
            feature_dict = dict(a_fs, **feature_dict)

            #dot to dest
            dot_to_dest = toward_dest(dots, sample[2])#8
            feature_dict = dict(dot_to_dest, **feature_dict)

            #angle changes along the path
            angle_changes = get_angle_change(dots)#4
            feature_dict = dict(angle_changes, **feature_dict)

            #other features
            other_features = get_other_features(dots)#1
            feature_dict = dict(other_features, **feature_dict)

            feature_dict = collections.OrderedDict(feature_dict)

            with codecs.open(prefix + 'feature_map', 'w', 'utf-8') as f_feature_map:
                #one line a feature name
                #the first line is the number of the features
                f_feature_map.write(str(len(feature_dict.keys())) + '\n')
                keys_list = feature_dict.keys()
                for keyi in keys_list:
                    f_feature_map.write(keyi + '\n')

            if(v_fs == {}) or (a_fs == {}):
                f3.write('%s\n'%(ID))
                continue
            features = ""
            for i,j in enumerate(feature_dict.values()):
                features = features + str(i) + ':' + str(j) + ' '
            f.write('%s %s\n'%(label, features))
            f2.write('%s\n'%(ID))
