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
import scipy as sp
import scipy.spatial
import scipy.stats
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


def get_dis(dots, axis):
    dots_list = []
    for i in range(len(dots)):
        dots_list.append(dots[i][axis])
    dots_list = np.array(dots_list)

    #add the feture to the dictionary
    feature_dic = {}
    feature_dic['point_coordinate_mean' + str(axis)] = dots_list.mean()
    feature_dic['point_coordinate_var' + str(axis)] = dots_list.var()
    feature_dic['point_coordinate_min' + str(axis)] = dots_list.min()
    feature_dic['point_coordinate_max' + str(axis)] = dots_list.max()
    # feature_dic['point_coordinate_range' + str(axis)] = \
    #     dots_list.max() - dots_list.min()
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
        # feature_dic['velocity_max' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = '$'
        feature_dic['velocity_z_per' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = '$'
        # feature_dic['velocity_v_num' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = '$'
        # feature_dic['velocity_zero_v_time_per' + str('_x_only_') + str(x_only) + str('_y_only_') + str( y_only)] = '$'
        feature_dic['velocity_pos' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = '$'
        # feature_dic['v_g_mean' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = '$'
        return feature_dic
    z_per = float(sum([1 for i in v if i == 0])) / len(v)
    z_v_time = 0
    for i in range(len(dots) - 1):
        if v[i] < eps:
            z_v_time += dots[i+1][2] - dots[i][2]
    zero_v_time_per = z_v_time / (eps + dots[-1][2] - dots[0][2])

    v_num = len(v)
    v = np.array(v)

    # add the feature to the dictionary
    feature_dic = {}
    feature_dic['velocity_mean' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = v.mean()
    feature_dic['velocity_var' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = v.var()
    feature_dic['velocity_min' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = v.min()
    # feature_dic['velocity_max' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = v.max()
    feature_dic['velocity_z_per' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = z_per
    # feature_dic['velocity_v_num' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = v_num
    # feature_dic['velocity_zero_v_time_per' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = zero_v_time_per
    feature_dic['velocity_init' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = v[0]
    # feature_dic['velocity_pos' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = np.argmax(v) / len(v)
    return feature_dic


def get_acc_speed(dots, x_only=False, y_only=False):
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
        # feature_dic['acc_g_mean' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = '$'
        return feature_dic
    acc = []
    for i in range(len(v) - 1):
        acc.append((v[i + 1] - v[i]) / (eps + dots[i + 1][2] - dots[i][2]))
        # acc.append((dots[i + 1][2] - dots[i][2]) / (eps + v[i + 1] - v[i]))

    z_per_acc = float(sum([1 for i in acc if i == 0])) / len(acc)
    v_num_acc = len(acc)

    acc = np.array(acc)

    # add the feature to the dictionary
    feature_dic = {}
    feature_dic['acc_mean' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = acc.mean()
    feature_dic['acc_max' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = acc.max()
    feature_dic['acc_min' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = acc.min()
    feature_dic['acc_var' + str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = acc.var()
    feature_dic['z_pre_acc'] = z_per_acc
    feature_dic['v_num_acc'] = v_num_acc
    feature_dic['acc_init'+ str('_x_only_') + str(x_only) + str('_y_only_') + str(y_only)] = acc[0]

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
    """
    result = []
    dist_to_dest = []
    for i in range(len(dots)):
        dist_to_dest.append((dots[i][0] - dest_point[0], dots[i][1] - dest_point[1]))

    tmp = []
    for i in range(len(dist_to_dest)):
        tmp.append(dist(dist_to_dest[i], [0, 0]))
    tmp = np.array(tmp)
    feature_dic = {}
    if len(tmp):
        # add the feature to the dictionary
        # feature_dic['toward_dist_mean'] = tmp.mean()
        feature_dic['toward_dist_min'] = tmp.min()
        feature_dic['toward_dist_max'] = tmp.max()
        # feature_dic['toward_dist_var'] = tmp.var()
    else:
        # feature_dic['toward_dist_mean'] = '$'
        feature_dic['toward_dist_min'] = '$'
        feature_dic['toward_dist_max'] = '$'
        # feature_dic['toward_dist_var'] = '$'

    aim_angle = np.array([math.atan(dist_to_dest[i][1] / (eps + dist_to_dest[i][0])) for i in range(len(dots))])
    if len(dots) > 1:
        aim_angle_diff = np.diff(aim_angle)
        feature_dic['aim_angle_diff_max'] = aim_angle_diff.max()
        feature_dic['aim_angle_diff_var'] = aim_angle_diff.var()
    else:
        feature_dic['aim_angle_diff_max'] = '$'
        feature_dic['aim_angle_diff_var'] = '$'
    return feature_dic


def get_other_features(dots):
    """
    Features that cannot be classify temporarily
    Including:
    1. If or not the dots in x axis go back out. 1-Y 0-N
    2. The density of x dots
    3. The count of pause of mouse trace
    4. x_init, y_init
    """
    x_dot = []
    y_dot = []
    for i in range(len(dots)):
        x_dot.append(dots[i][0])
        y_dot.append(dots[i][1])
    x_dot = np.array(x_dot)
    y_dot = np.array(y_dot)
    x_back_num = 0
    if len(dots) > 1:
        x_back_num = (np.diff(x_dot) < 0).sum()

    density_0 = get_density(dots, 0)
    density_1 = get_density(dots, 1)
    density_2 = get_density(dots, 2)

    pause = 0

    for i in range(len(dots) - 1):
        if dots[i + 1][0] == dots[i][0]:
            pause += 1

    return {'go_back_x':x_back_num, 'density_2': density_2, 'density_0': density_0,
            'density_1': density_1, 'pause': pause, 'x_init': dots[0][0], 'y_init': dots[0][1]}


def get_density(dots, axis):
    """
    Features related to the density of points about x-ray
    If there is only one point return 0. If not, return density.
    """
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
    feature_dic['smooth_1_' + str(axis)] = smooth_1
    return feature_dic


def get_y_min(dots):
    """
    Features related to y-values.
    Return the minimum of y-values.
    """
    y = []
    for i in dots:
        y.append(int(i[1]))
    y = np.array(y)
    return [y.min()]


def get_ab_direction(feature_dic, dots):
    """
    A, B, C ...
    the angle between AB and the horizontal line
    """
    point_minus = []
    for i in range(len(dots) - 1):
        point_minus.append((dots[i+1][0] - dots[i][0], dots[i+1][1] - dots[i][1]))
    ab_angle = []
    if len(point_minus):
        for i in range(len(point_minus)):
            angle_i = math.atan(point_minus[i][1] / (eps + point_minus[i][0]))
            ab_angle.append(angle_i)
        ab_angle = np.array(ab_angle)

        feature_dic['ab_angle_mean'] = ab_angle.mean()
        feature_dic['ab_angle_min'] = ab_angle.min()
        feature_dic['ab_angle_max'] = ab_angle.max()
        feature_dic['ab_angle_var'] = ab_angle.var()
        feature_dic['ab_angle_range'] = ab_angle.max() - ab_angle.min()
        feature_dic['ab_angle_kurt'] = sp.stats.kurtosis(ab_angle)
    else:
        feature_dic['ab_angle_mean'] = '$'
        feature_dic['ab_angle_min'] = '$'
        feature_dic['ab_angle_max'] = '$'
        feature_dic['ab_angle_var'] = '$'
        feature_dic['ab_angle_range'] = '$'
        feature_dic['ab_angle_kurt'] = '$'

def get_angle_change(dots):
    """
    A, B, C
    get the angle of BA and BC
    """
    point_minus = []
    for i in range(len(dots) - 1):
        point_minus.append((dots[i+1][0] - dots[i][0], dots[i+1][1] - dots[i][1]))

    tmp = []
    if len(point_minus) == 1:
        tmp.append(0)
    else:
        for i in range(len(point_minus) - 1):
            abc_angle = dotProduct(point_minus[i], (-point_minus[i+1][0], -point_minus[i+1][1])) / (eps +
                                        dist(point_minus[i], [0, 0]) * dist(point_minus[i+1], [0, 0]))
            tmp.append(math.acos(abc_angle))
    feature_dic = {}
    tmp = np.array(tmp)
    # add the feture to the dictionary
    if len(tmp):
        # feature_dic['angle_mean'] = tmp.mean()
        # feature_dic['angle_min'] = tmp.min()
        # feature_dic['angle_max'] = tmp.max()
        feature_dic['angle_var'] = tmp.var()
        feature_dic['angle_kurt_angle'] = sp.stats.kurtosis(tmp)
    else:
        # feature_dic['angle_mean'] = '$'
        # feature_dic['angle_min'] = '$'
        # feature_dic['angle_max'] = '$'
        feature_dic['angle_var'] = '$'
        feature_dic['angle_kurt_angle'] = '$'

    angle_v = []
    # get angle velocity
    angle_change = list(tmp)
    for i in range(len(angle_change) - 1):
        angle_v.append((angle_change[i+1] - angle_change[i]) / (eps + (dots[i+1][2] - dots[i][2])))
    angle_v = np.array(angle_v)
    if len(angle_v) > 0:
        feature_dic['angle_v_mean'] = angle_v.mean()
        # feature_dic['angle_v_min'] = angle_v.min()
        # feature_dic['angle_v_max'] = angle_v.max()
        feature_dic['angle_v_var'] = angle_v.var()
    else:
        feature_dic['angle_v_mean'] = '$'
        # feature_dic['angle_v_min'] = '$'
        # feature_dic['angle_v_max'] = '$'
        feature_dic['angle_v_var'] = '$'

    # get angle acc
    angle_acc = []
    angle_v = list(angle_v)
    if len(angle_v) > 1:
        for i in range(len(angle_v) - 1):
            angle_acc.append((angle_v[i + 1] - angle_v[i]) / (eps + (dots[i + 1][2] - dots[i][2])))
        angle_acc = np.array(angle_acc)
        # feature_dic['angle_ac_mean'] = angle_acc.mean()
        # feature_dic['angle_acc_min'] = angle_acc.min()
        feature_dic['angle_acc_max'] = angle_acc.max()
        feature_dic['angle_acc_var'] = angle_acc.var()
    else:
        # feature_dic['angle_ac_mean'] = '$'
        # feature_dic['angle_acc_min'] = '$'
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
    time_zero = 0
    if len(dots) > 1:
        for i in range(len(dots) - 1):
            if dots[i][0] == dots[i+1][0]:
                time_zero += dots[i+1][2] - dots[i][2]
        time_pause_radio = time_zero / (eps + time_list.max() - time_list.min())
        feature_dic['time_pause_radio'] = time_pause_radio
    else:
        feature_dic['time_pause_radio'] = '$'
    # feature_dic['time_var'] = time_list.var()

    time_interval = []
    for i in range(len(time_list) - 1):
        time_interval.append(time_list[i+1] - time_list[i])

    time_interval = np.array(time_interval)
    if len(time_interval):
        feature_dic['time_interval_mean'] = time_interval.mean()
        feature_dic['time_interval_min'] = time_interval.min()
        # feature_dic['time_interval_max'] = time_interval.max()
        feature_dic['time_interval_var'] = time_interval.var()
    else:
        feature_dic['time_interval_mean'] = '$'
        feature_dic['time_interval_min'] = '$'
        # feature_dic['time_interval_max'] = '$'
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
    length_sum = 0
    for i in range(len(dots) - 1):
        length_sum += math.sqrt(pow_2(dots[i+1][1] - dots[i][1]) + pow_2(dots[i+1][0] - dots[i][0]))
    feature_dic['dots_num'] = len(dots)
    feature_dic['straight'] = sp.spatial.distance.euclidean((dots[0][0], dots[0][1]), (dots[-1][0], dots[-1][1]))/(eps +
                                                                                                        length_sum)
    # negative features
    # TCM = 0
    # for i in range(len(dots) - 1):
    #     TCM += (dots[i + 1][2] - dots[i][2]) * math.sqrt(pow_2(dots[i][0] - dots[i+1][0]) +
    #                                                      pow_2(dots[i][1] - dots[i+1][1]))
    # TCM /= (length_sum + eps)
    #
    # SC = -TCM * TCM
    # for i in range(len(dots) - 1):
    #     SC += pow_2(dots[i + 1][2] - dots[i][2]) * math.sqrt(pow_2(dots[i][0] - dots[i+1][0]) +
    #                                                          pow_2(dots[i][1] - dots[i+1][1]))
    # feature_dic['TCM'] = TCM
    # feature_dic['SC'] = SC
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
        feature_dic['h_angle_kurt_angle'] = sp.stats.kurtosis(h_angle_degree)
        # feature_dic['h_angle_mean'] = h_angle_degree.mean()
        # feature_dic['h_angle_min'] = h_angle_degree.min()
        feature_dic['h_angle_max'] = h_angle_degree.max()
        feature_dic['h_angle_var'] = h_angle_degree.var()
        # feature_dic['h_angle_range'] = h_angle_degree.max() - h_angle_degree.min()
    else:
        feature_dic['h_angle_kurt_angle'] = '$'
        # feature_dic['h_angle_mean'] = '$'
        # feature_dic['h_angle_min'] = '$'
        feature_dic['h_angle_max'] = '$'
        feature_dic['h_angle_var'] = '$'
        # feature_dic['h_angle_range'] = '$'

    # if len(h_angle_degree) > 1:
    #     feature_dic['smooth_h_angle'] = np.std(np.diff(h_angle_degree)) / (eps + abs((np.diff(h_angle_degree)).mean()))
    # else:
    #     feature_dic['smooth_h_angle'] = '$'

    h_angle_speed = []
    for i in range(len(h_angle_degree) - 1):
        h_angle_speed.append((h_angle_degree[i + 1] - h_angle_degree[i]) / (eps + (dots[i + 1][2] - dots[i][2])))

    h_angle_speed = np.array(h_angle_speed)
    if len(h_angle_speed):
        # feature_dic['h_angle_speed_mean'] = h_angle_speed.mean()
        # feature_dic['h_angle_speed_min'] = h_angle_speed.min()
        # feature_dic['h_angle_speed_max'] = h_angle_speed.max()
        feature_dic['h_angle_speed_var'] = h_angle_speed.var()
    else:
        # feature_dic['h_angle_speed_mean'] = '$'
        # feature_dic['h_angle_speed_min'] = '$'
        # feature_dic['h_angle_speed_max'] = '$'
        feature_dic['h_angle_speed_var'] = '$'

    # h_angle_acc = []
    # for i in range(len(h_angle_speed) - 1):
    #     h_angle_acc.append((h_angle_speed[i + 1] - h_angle_speed[i]) / (eps + (dots[i + 1][2] - dots[i][2])))
    #
    # h_angle_acc = np.array(h_angle_acc)
    # if len(h_angle_acc):
    #     feature_dic['h_angle_acc_mean'] = h_angle_acc.mean()
    #     feature_dic['h_angle_acc_min'] = h_angle_acc.min()
    #     feature_dic['h_angle_acc_max'] = h_angle_acc.max()
    #     feature_dic['h_angle_acc_var'] = h_angle_acc.var()
    # else:
    #     feature_dic['h_angle_acc_mean'] = '$'
    #     feature_dic['h_angle_acc_min'] = '$'
    #     feature_dic['h_angle_acc_max'] = '$'
    #     feature_dic['h_angle_acc_var'] = '$'

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
            ac_length = sp.spatial.distance.euclidean((dots[i-1][0], dots[i-1][1]), (dots[i+1][0], dots[i+1][1]))
            cur_dis.append(point_to_line(dots[i], dots[i-1], dots[i+1]) / (ac_length + eps))

        cur_dis = np.array(cur_dis)
        feature_dic['curvature_distance_mean'] = cur_dis.mean()
        feature_dic['curvature_distance_min'] = cur_dis.min()
        feature_dic['curvature_distance_max'] = cur_dis.max()
        feature_dic['curvature_distance_var'] = cur_dis.var()
        feature_dic['curvature_distance_range'] = cur_dis.max() - cur_dis.min()
        # feature_dic['curvature_dis_kurt_angle'] = sp.stats.kurtosis(cur_dis)
    else:
        feature_dic['curvature_distance_mean'] = '$'
        feature_dic['curvature_distance_min'] = '$'
        feature_dic['curvature_distance_max'] = '$'
        feature_dic['curvature_distance_var'] = '$'
        feature_dic['curvature_distance_range'] = '$'
        # feature_dic['curvature_dis_kurt_angle'] = '$'

    return feature_dic


def get_curvature_rate(feature_dic, dots):
    """
    curvature rate and curvature rate diff
    """
    c_angle_degree = []
    l_dis = []
    for i in range(len(dots)):
        c_angle_degree.append(math.atan(dots[i][1] / (eps + dots[i][0])))
        # l_dis.append(sp.spatial.distance.euclidean((dots[i][0], dots[i][1]), (0, 0)))

    c_cu = []
    if len(dots) > 1:
        for i in range(len(dots) - 1):
            c_cu.append((c_angle_degree[i + 1] - c_angle_degree[i]) / (eps +
                                        sp.spatial.distance.euclidean((dots[i + 1][0], dots[i + 1][1]),
                                                                      (dots[i][0], dots[i][1]))))

        c_cu = np.array(c_cu)

        # feature_dic['curvature_rate_med'] = np.median(c_cu)
        feature_dic['curvature_rate_mean'] = c_cu.mean()
        feature_dic['curvature_rate_min'] = c_cu.min()
        feature_dic['curvature_rate_max'] = c_cu.max()
        feature_dic['curvature_rate_range'] = c_cu.max() - c_cu.min()
        feature_dic['curvature_rate_var'] = c_cu.var()
        feature_dic['curvature_rate_kurt_angle'] = sp.stats.kurtosis(c_cu)
        if len(c_cu) > 1:
            smooth_curvature = np.std(np.diff(c_cu)) / (eps + abs((np.diff(c_cu)).mean()))
            curvature_diff = np.diff(c_cu)
            feature_dic['curvature_rate_diff_mean'] = curvature_diff.mean()
            # feature_dic['curvature_rate_diff_min'] = curvature_diff.min()
            # feature_dic['curvature_rate_diff_max'] = curvature_diff.max()
            feature_dic['curvature_rate_diff_range'] = curvature_diff.max() - curvature_diff.min()
            feature_dic['curvature_rate_diff_var'] = curvature_diff.var()
            # feature_dic['curvature_rate_smooth_var'] = smooth_curvature
            # feature_dic['curvature_rate_zero_var'] = len(curvature_diff[curvature_diff == 0])
        else:
            # feature_dic['curvature_rate_diff_mean'] = '$'
            feature_dic['curvature_rate_diff_min'] = '$'
            # feature_dic['curvature_rate_diff_max'] = '$'
            feature_dic['curvature_rate_diff_range'] = '$'
            feature_dic['curvature_rate_diff_var'] = '$'
            # feature_dic['curvature_rate_smooth_var'] = '$'
            # feature_dic['curvature_rate_zero_var'] = '$'
    else:
        # feature_dic['curvature_rate_med'] = '$'
        feature_dic['curvature_rate_mean'] = '$'
        feature_dic['curvature_rate_min'] = '$'
        feature_dic['curvature_rate_max'] = '$'
        feature_dic['curvature_rate_range'] = '$'
        feature_dic['curvature_rate_var'] = '$'
        feature_dic['curvature_rate_kurt_angle'] = '$'


def extract_features(file, with_label=True, prefix=''):
    """
    Extract features and save features in LibSVM format
    Input: dataset filename
    Output: data Id && features  
    v_fs = velocity features
    a_fs = accerlation features
    
    """
    f = open(prefix+'sample-features', 'w')
    f2 = open(prefix+'id-map', 'w')
    f3 = open(prefix+'inval-id', 'w')
    with codecs.open(file, 'r', 'utf-8') as fdata:
        for line in fdata.readlines():
            line = line.strip()
            sample = handle_one(line, with_label=with_label)

            # sort the dots according to ascending time order
            # it seems labels of dots in the descending order time is all 1(not rebot), use this part data
            dots = np.array(sample[1])
            if len(dots) < 4:
                continue
            dots = sorted(dots, key=lambda x: x[2])
            for i in range(len(dots)):
                dots[i] = list(dots[i])

            ID = sample[0]
            label = sample[3]

            feature_dict = dict()
            # point coordinate features
            point_coordx_f = get_dis(dots, 0)
            feature_dict = dict(point_coordx_f, **feature_dict)

            point_coordy_f = get_dis(dots, 1)
            feature_dict = dict(point_coordy_f, **feature_dict)

            # velocity features
            v_fs = get_velocity(dots)
            feature_dict = dict(v_fs, **feature_dict)

            v_fs_x_only = get_velocity(dots, x_only=True)
            feature_dict = dict(v_fs_x_only, **feature_dict)

            v_fs_y_only = get_velocity(dots, y_only=True)
            feature_dict = dict(v_fs_y_only, **feature_dict)

            # acc features
            a_fs_y_only = get_acc_speed(dots, y_only=True)
            feature_dict = dict(a_fs_y_only, **feature_dict)

            a_fs_x_only = get_acc_speed(dots, x_only=True)
            feature_dict = dict(a_fs_x_only, **feature_dict)

            a_fs = get_acc_speed(dots)
            feature_dict = dict(a_fs, **feature_dict)

            # dot to dest
            dot_to_dest = toward_dest(dots, sample[2])
            feature_dict = dict(dot_to_dest, **feature_dict)

            # angle changes, v, acc along the path
            angle_changes = get_angle_change(dots)
            feature_dict = dict(angle_changes, **feature_dict)
            get_ab_direction(feature_dict, dots)

            # get_horizon_angle
            feature_dict = dict(get_horizon_angle(dots), **feature_dict)

            # get curvature_distance in the trace
            feature_dict = dict(curvature_distance(dots), **feature_dict)
            get_curvature_rate(feature_dict, dots)

            # other features
            other_features = get_other_features(dots)
            feature_dict = dict(other_features, **feature_dict)

            # time series features
            time_features = get_time_feature(dots)
            feature_dict = dict(time_features, **feature_dict)

            # length of the dots
            feature_dict = dict(get_length_dots(dots), **feature_dict)

            # Smooth
            for i in range(3):
                smooth_feature = get_smooth(dots, i)
                feature_dict = dict(smooth_feature, **feature_dict)

            feature_dict = collections.OrderedDict(feature_dict)

            with codecs.open(prefix + 'feature_map', 'w', 'utf-8') as f_feature_map:
                # one line a feature name
                # the first line is the number of the features
                f_feature_map.write(str(len(feature_dict.keys())) + '\n')
                keys_list = feature_dict.keys()
                for keyi in keys_list:
                    f_feature_map.write(keyi + '\n')

            if(v_fs == {}) or (a_fs == {}):
                f3.write('%s\n'%(ID))
                continue
            features = ""
            for i, j in enumerate(feature_dict.values()):
                if str(j) == '$':
                    continue
                features = features + str(i) + ':' + str(j) + ' '
            f.write('%s %s\n'%(label, features))
            f2.write('%s\n' %ID)
