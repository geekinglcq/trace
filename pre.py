from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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
    feature_dic['point_coordinate_min' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = dots_list.min()
    feature_dic['point_coordinate_max' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = dots_list.max()
    return feature_dic

def get_velocity(dots, x_only=False, y_only=False):
    """
    Get the features that related to acceleration
    Input: dots series, x_only flag to control if we only take x axis as consideration 
    Return: a list of features related to velocity, including [mean, max, min, variance, z_per]
    z_per -- denote the percentage of zero in whole velocity list
    v_num -- volumn of velocity points
    zero_v_time_per -- zero velocity's percentage in the all time
    if no velocity can be calculated, return None

    #process the missing values(using a special char and than ignore it, xgboost can process the missing value
    """
    
    v = []
    for i in range(len(dots) - 1):
        v.append(float(dist(dots[i + 1], dots[i], x_only=x_only, y_only=y_only)) / (eps + dots[i + 1][2] - dots[i][2]))
    if len(v) ==0:
        feature_dic = {}
        feature_dic['velocity_mean' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = '$'
        feature_dic['velocity_var' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = '$'
        feature_dic['velocity_min' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = '$'
        feature_dic['velocity_max' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = '$'
        feature_dic['velocity_z_per' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = '$'
        feature_dic['velocity_v_num' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = '$'
        feature_dic['velocity_zero_v_time_per' + str('_x_only_') + str(x_only) + str('_y_only_') + str( y_only)] = '$'
        return feature_dic
    z_per = float(sum([1 for i in v if i == 0])) / len(v)
    z_v_time = 0
    for i in range(len(dots) - 1):
        if v[i] < eps:
            z_v_time += dots[i+1][2] - dots[i][2]
    zero_v_time_per = z_v_time / (eps + dots[-1][2] - dots[0][2])

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
    feature_dic['velocity_zero_v_time_per' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = zero_v_time_per
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
        feature_dic = {}
        feature_dic['acc_mean' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = '$'
        feature_dic['acc_max' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = '$'
        feature_dic['acc_min' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = '$'
        feature_dic['acc_var' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = '$'
        feature_dic['z_pre_acc'] = '$'
        feature_dic['v_num_acc'] = '$'
        return feature_dic
    acc = []
    for i in range(len(v) - 1):
        acc.append((v[i + 1] - v[i]) / (eps + dots[i + 1][2] - dots[i][2]))

    z_per_acc = float(sum([1 for i in acc if i == 0])) / len(acc)
    v_num_acc = len(acc)

    acc = np.array(acc)

    # add the feture to the dictionary
    feature_dic = {}
    feature_dic['acc_mean' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = acc.mean()
    feature_dic['acc_max' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = acc.max()
    feature_dic['acc_min' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = acc.min()
    feature_dic['acc_var' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = acc.var()
    feature_dic['z_pre_acc'] = z_per_acc
    feature_dic['v_num_acc'] = v_num_acc
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
    else:
        feature_dic['toward_dist_mean'] = '$'
        feature_dic['toward_dist_min'] = '$'
        feature_dic['toward_dist_max'] = '$'
        feature_dic['toward_dist_var'] = '$'

    tmp = []
    for i in range(len(dist_to_dest) - 1):
        tmp.append(dotProduct(dist_to_dest[i], dist_to_dest[i+1]))
    tmp = np.array(tmp)
    if len(tmp):
        feature_dic['toward_dist_dot_product_mean'] = tmp.mean()
        feature_dic['toward_dist_dot_product_min'] = tmp.min()
        feature_dic['toward_dist_dot_product_max'] = tmp.max()
        feature_dic['toward_dist_dot_product_var'] = tmp.var()
    else:
        feature_dic['toward_dist_dot_product_mean'] = '$'
        feature_dic['toward_dist_dot_product_min'] = '$'
        feature_dic['toward_dist_dot_product_max'] = '$'
        feature_dic['toward_dist_dot_product_var'] = '$'
    return feature_dic

def get_other_features(dots):
    """
    Features that cannot be classify temporarily
    Including:
    1. If or not the dots in x axis go back out. 1-Y 0-N
    2. The density of x dots
    3. The count of pause of mouse trace
    """
    go_back = 0
    for i in range(len(dots) - 1):
        if (dots[i + 1][0] < dots[i][0]):
            go_back = 1
    
    
    density_0 = get_density(dots, 0)
    density_1 = get_density(dots, 1)

    pause = 0
    for i in range(len(dots) - 1):
        if (dots[i + 1][0] == dots[i][0]):
            pause += 1


    return {'go_back':go_back, 'density_0': density_0, 'density_1': density_1, 'pause': pause}

def get_density(dots, axis):
    '''
    Features related to the density of points about x-ray
    If there is only one point return 0. If not, return density.
    '''
    x = []
    for i in dots:
        x.append(int(i[axis]))
    x = np.array(x)
    if (x.max()-x.min())>=1:
        density = float(len(x))/(x.max()-x.min())
    else:
        density = 0
    return density


def get_smooth(dots, axis):
    """
    measure smoothness of a time series
    """
    dots_axis = []
    for i in range(len(dots)):
        dots_axis.append(dots[i][axis])
    dots_axis = np.array(dots_axis)
    smooth_1 = 0
    if len(dots) > 1:
        smooth_1 = np.std(np.diff(dots_axis)) / (eps + abs((np.diff(dots_axis)).mean()))
    feature_dic = {}
    feature_dic['*smooth_1_' + str(axis)] = smooth_1

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
    else:
        feature_dic['angle_mean'] = '$'
        feature_dic['angle_min'] = '$'
        feature_dic['angle_max'] = '$'
        feature_dic['angle_var'] = '$'

    angle_v = []
    #get angle velocity
    angle_change = list(tmp)
    for i in range(len(angle_change) - 1):
        angle_v.append((angle_change[i+1] - angle_change[i]) / (eps + (dots[i+1][2] - dots[i][2])))
    angle_v = np.array(angle_v)
    if len(angle_v) > 0:
        feature_dic['angle_v_mean'] = angle_v.mean()
        feature_dic['angle_v_min'] = angle_v.min()
        feature_dic['angle_v_max'] = angle_v.max()
        feature_dic['angle_v_var'] = angle_v.var()
    else:
        feature_dic['angle_v_mean'] = '$'
        feature_dic['angle_v_min'] = '$'
        feature_dic['angle_v_max'] = '$'
        feature_dic['angle_v_var'] = '$'

    #get angle acc
    angle_acc = []
    angle_v = list(angle_v)
    if len(angle_v) > 1:
        for i in range(len(angle_v) - 1):
            angle_acc.append((angle_v[i + 1] - angle_v[i]) / (eps + (dots[i + 1][2] - dots[i][2])))
        angle_acc = np.array(angle_acc)
        feature_dic['angle_ac_mean'] = angle_acc.mean()
        feature_dic['angle_acc_min'] = angle_acc.min()
        feature_dic['angle_acc_max'] = angle_acc.max()
        feature_dic['angle_acc_var'] = angle_acc.var()
    else:
        feature_dic['angle_ac_mean'] = '$'
        feature_dic['angle_acc_min'] = '$'
        feature_dic['angle_acc_max'] = '$'
        feature_dic['angle_acc_var'] = '$'

    return feature_dic

def get_time_feature(dots):
    """
    get features related to time series(time only)
    """
    feature_dic = {}
    time_list = []
    for i in range(len(dots)):
        time_list.append(dots[i][2])
    time_list = np.array(time_list)

    feature_dic['time_mean'] = time_list.mean()
    feature_dic['time_duration'] = time_list.max() - time_list.min()
    feature_dic['time_var'] = time_list.var()

    time_interval = []
    for i in range(len(time_list) - 1):
        time_interval.append(time_list[i+1] - time_list[i])

    time_interval = np.array(time_interval)
    if len(time_interval):
        feature_dic['time_interval_mean'] = time_interval.mean()
        feature_dic['time_interval_min'] = time_interval.min()
        feature_dic['time_interval_max'] = time_interval.max()
        feature_dic['time_interval_var'] = time_interval.var()
    else:
        feature_dic['time_interval_mean'] = '$'
        feature_dic['time_interval_min'] = '$'
        feature_dic['time_interval_max'] = '$'
        feature_dic['time_interval_var'] = '$'

    return feature_dic

def pow_2(x):
    return x * x
def get_length_dots(dots):
    """
    1. the number of points of the dots
    2. the length of the dots
    """
    feature_dic = {}
    sum = 0
    for i in range(len(dots) - 1):
        sum += math.sqrt(pow_2(dots[i+1][1] - dots[i][1]) + pow_2(dots[i+1][0] - dots[i][0]))
    feature_dic['dots_num'] = len(dots)
    feature_dic['dots_length'] = len(dots)

    return feature_dic
def get_horizon_angle(dots):
    """
    For consecutive points A, B: angle between
    line AB and horizontal line
    1. h_angle change
    2. h_angle_speed
    3. h_angle_acc
    """
    point_minus = []
    for i in range(len(dots) - 1):
        point_minus.append(dotMinus(dots[i + 1], dots[i]))

    h_angle_degree = []
    for i in range(len(point_minus)):
        if point_minus[i] == 0:
            h_angle_degree.append(1.57075)
        else:
            h_angle_degree.append(math.atan(point_minus[i][1] / (eps + point_minus[i][0])))
    h_angle_degree = np.array(h_angle_degree)
    feature_dic = {}
    if len(h_angle_degree):
        feature_dic['h_angle_mean'] = h_angle_degree.mean()
        feature_dic['h_angle_min'] = h_angle_degree.min()
        feature_dic['h_angle_max'] = h_angle_degree.max()
        feature_dic['h_angle_var'] = h_angle_degree.var()
    else:
        feature_dic['h_angle_mean'] = '$'
        feature_dic['h_angle_min'] = '$'
        feature_dic['h_angle_max'] = '$'
        feature_dic['h_angle_var'] = '$'

    h_angle_speed = []
    for i in range(len(h_angle_degree) - 1):
        h_angle_speed.append((h_angle_degree[i + 1] - h_angle_degree[i]) / (eps + (dots[i + 1][2] - dots[i][2])))

    h_angle_speed = np.array(h_angle_speed)
    if len(h_angle_speed):
        feature_dic['h_angle_speed_mean'] = h_angle_speed.mean()
        feature_dic['h_angle_speed_min'] = h_angle_speed.min()
        feature_dic['h_angle_speed_max'] = h_angle_speed.max()
        feature_dic['h_angle_speed_var'] = h_angle_speed.var()
    else:
        feature_dic['h_angle_speed_mean'] = '$'
        feature_dic['h_angle_speed_min'] = '$'
        feature_dic['h_angle_speed_max'] = '$'
        feature_dic['h_angle_speed_var'] = '$'

    h_angle_acc = []
    for i in range(len(h_angle_speed) - 1):
        h_angle_acc.append((h_angle_speed[i + 1] - h_angle_speed[i]) / (eps + (dots[i + 1][2] - dots[i][2])))

    h_angle_acc = np.array(h_angle_acc)
    if len(h_angle_acc):
        feature_dic['h_angle_acc_mean'] = h_angle_acc.mean()
        feature_dic['h_angle_acc_min'] = h_angle_acc.min()
        feature_dic['h_angle_acc_max'] = h_angle_acc.max()
        feature_dic['h_angle_acc_var'] = h_angle_acc.var()
    else:
        feature_dic['h_angle_acc_mean'] = '$'
        feature_dic['h_angle_acc_min'] = '$'
        feature_dic['h_angle_acc_max'] = '$'
        feature_dic['h_angle_acc_var'] = '$'

    return feature_dic

def point_to_line(data1, data2, p):
    """
    get the dis of a point to a line
    """
    a = data1[1] - data2[1]
    b = data2[0] - data1[0]
    c = data1[0] * data2[1] - data1[1] * data2[0]
    td = (a * p[0] + b * p[1] + c) / (eps + math.sqrt(a * a + b * b))
    return td
def curvature_distance(dots):
    """
    A, B, C in the trace
    get the dis of C to line AB
    """
    feature_dic = {}
    cur_dis = []
    if len(dots) >= 3:
        for i in range(1, len(dots) - 1):
            cur_dis.append(point_to_line(dots[i], dots[i-1], dots[i+1]))

        cur_dis = np.array(cur_dis)
        feature_dic['curvature_distance_mean'] = cur_dis.mean()
        feature_dic['curvature_distance_min'] = cur_dis.min()
        feature_dic['curvature_distance_max'] = cur_dis.max()
        feature_dic['curvature_distance_var'] = cur_dis.var()
    else:
        feature_dic['curvature_distance_mean'] = '$'
        feature_dic['curvature_distance_min'] = '$'
        feature_dic['curvature_distance_max'] = '$'
        feature_dic['curvature_distance_var'] = '$'

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

            #angle changes, v, acc along the path
            angle_changes = get_angle_change(dots)#4
            feature_dict = dict(angle_changes, **feature_dict)

            #get_horizon_angle
            feature_dict = dict(get_horizon_angle(dots), **feature_dict) #12

            #get curvature_distance in the trace #4
            feature_dict = dict(curvature_distance(dots), **feature_dict)  #4

            #other features
            other_features = get_other_features(dots)#2
            feature_dict = dict(other_features, **feature_dict)

            #time series features
            time_features = get_time_feature(dots)#7
            feature_dict = dict(time_features, **feature_dict)

            #length of the dots
            feature_dict = dict(get_length_dots(dots), **feature_dict)#2

            #Smooth
            for i in range(1):
                smooth_feature = get_smooth(dots, i)
                feature_dict = dict(smooth_feature, **feature_dict)

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
                if str(j) == '$':
                    continue
                features = features + str(i) + ':' + str(j) + ' '
            f.write('%s %s\n'%(label, features))
            f2.write('%s\n'%(ID))
