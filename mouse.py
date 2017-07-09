import pandas as pd
import scipy as sp
import numpy as np
import sklearn
import gc
import warnings
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
import matplotlib
import os

warnings.filterwarnings("ignore")
import math

eps = 1e-6
cache = 'cache'
sub = 'sub'
datadir = 'data'

train_path = os.path.join(datadir, 'new_train.txt')
test_path = os.path.join(datadir, 'dsjtzs_txfz_test1.txt')

if not os.path.exists(cache):
    os.mkdir(cache)
if not os.path.exists(sub):
    os.mkdir(sub)


def applyParallel(dfGrouped, func):
    with Parallel(n_jobs=40) as parallel:
        retLst = parallel(delayed(func)(pd.Series(value)) for key, value in dfGrouped)
        return pd.concat(retLst, axis=0)


def draw(df):
    import matplotlib.pyplot as plt
    if not os.path.exists('pic'):
        os.mkdir('pic')

    points = []
    for point in df.trajectory[:-1].split(';'):
        point = point.split(',')
        points.append((float(point[0]) / 7, float(point[1]) / 13))

    x, y = zip(*points)
    plt.figure(figsize=(16, 5))
    plt.subplot(121)
    plt.plot(x, y)
    plt.subplot(122)
    plt.plot(x, y)
    aim = df.aim.split(',')
    aim = (float(aim[0]) / 7, float(aim[1]) / 13)
    plt.scatter(aim[0], aim[1])
    plt.title(df.label)
    plt.savefig('pic/%s-label=%s' % (df.idx, df.label))
    plt.clf()
    plt.close()


# ***************add feature*******************
def dist(a, b, x_only=False, y_only=False):
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


def get_dis(feature_dic, dots, x_only=False, y_only=False):
    dots_list = []
    for i in range(len(dots)):
        dots_list.append(float(dist([0, 0], dots[i], x_only=x_only, y_only=y_only)))
    dots_list = np.array(dots_list)

    feature_dic['*p_axis_mean' + str('_x_') + str(x_only) + str('_y_') + str(y_only)] = dots_list.mean()
    feature_dic['*p_axis_var' + str('_x_') + str(x_only) + str('_y_') + str(y_only)] = dots_list.var()
    feature_dic['*p_axis_min' + str('_x_') + str(x_only) + str('_y_') + str(y_only)] = dots_list.min()
    feature_dic['*p_axis_max' + str('_x_') + str(x_only) + str('_y_') + str(y_only)] = dots_list.max()


def get_velocity(feature_dic, dots, x_only=False, y_only=False):
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
    z_per = float(sum([1 for i in v if i == 0])) / (eps + len(v))
    z_v_time = 0
    for i in range(len(dots) - 1):
        if v[i] < eps:
            z_v_time += dots[i + 1][2] - dots[i][2]
    zero_v_time_per = z_v_time / (eps + dots[-1][2] - dots[0][2])

    v_num = len(v)
    v = np.array(v)

    if len(v):
        # add the feture to the dictionary
        feature_dic['*v_median' + str('_x_') + str(x_only) + str('_y_') + str(y_only)] = np.median(v)
        feature_dic['*v_mean' + str('_x_') + str(x_only) + str('_y_') + str(y_only)] = v.mean()
        feature_dic['*v_var' + str('_x_') + str(x_only) + str('_y_') + str(y_only)] = v.var()
        feature_dic['*v_min' + str('_x_') + str(x_only) + str('_y_') + str(y_only)] = v.min()
        feature_dic['*v_max' + str('_x_') + str(x_only) + str('_y_') + str(y_only)] = v.max()
        feature_dic['*v_z_per' + str('_x_') + str(x_only) + str('_y_') + str(y_only)] = z_per
        feature_dic['*v_num' + str('_x_') + str(x_only) + str('_y_') + str(y_only)] = v_num
        feature_dic['*v_time_per' + str('_x_') + str(x_only) + str('_y_') + str(y_only)] = zero_v_time_per


def get_acc_speed(feature_dic, dots, x_only=False, y_only=False):
    """
    Get the features that related to acceleration
    Input: dots series, x_only flag to control if we only take x axis as consideration
    Return: a list of features related to velocity, including [mean, max, min, variance]
    if no acc-speed can be calculated, return None
    """
    v = []
    for i in range(len(dots) - 1):
        v.append(float(dist(dots[i + 1], dots[i], x_only=x_only, y_only=y_only)) / (eps + dots[i + 1][2] - dots[i][2]))
    if len(v) > 1:
        acc = []
        for i in range(len(v) - 1):
            acc.append((v[i + 1] - v[i]) / (eps + dots[i + 1][2] - dots[i][2]))

        z_per_acc = float(sum([1 for i in acc if i == 0])) / len(acc)
        v_num_acc = len(acc)

        acc = np.array(acc)

        # add the feture to the dictionary
        feature_dic['*acc_median' + str('_x_') + str(x_only) + str('_y_') + str(y_only)] = np.median(acc)
        feature_dic['*acc_mean' + str('_x_') + str(x_only) + str('_y_') + str(y_only)] = acc.mean()
        feature_dic['*acc_max' + str('_x_') + str(x_only) + str('_y_') + str(y_only)] = acc.max()
        feature_dic['*acc_min' + str('_x_') + str(x_only) + str('_y_') + str(y_only)] = acc.min()
        feature_dic['*acc_var' + str('_x_') + str(x_only) + str('_y_') + str(y_only)] = acc.var()
        feature_dic['*z_pre_acc'] = z_per_acc
        feature_dic['*v_num_acc'] = v_num_acc
        feature_dic['*acc_init'] = acc[0]
        feature_dic['*acc_last'] = acc[-1]


def get_density(df, dots, axis):
    '''
    Features related to the density of points about axis
    If there is only one point return 0. If not, return density.
    '''
    x = []
    for i in dots:
        x.append(int(i[axis]))
    x = np.array(x)
    if (x.max() - x.min()) >= 1:
        density = float(len(x)) / (x.max() - x.min())
    else:
        density = 0
    df['*density_' + str(axis)] = density


def point_to_line(data1, data2, p):
    """
    get the dis of a point to a line
    """
    a = data1[1] - data2[1]
    b = data2[0] - data1[0]
    c = data1[0] * data2[1] - data1[1] * data2[0]
    td = (a * p[0] + b * p[1] + c) / (eps + math.sqrt(a * a + b * b))
    return td


def curvature_distance(feature_dic, dots):
    """
    A, B, C in the trace
    get the dis of C to line AB
    """
    cur_dis = []
    if len(dots) >= 3:
        for i in range(1, len(dots) - 1):
            cur_dis.append(point_to_line(dots[i], dots[i - 1], dots[i + 1]))

        cur_dis = np.array(cur_dis)
        feature_dic['*curvature_dis_med'] = np.median(cur_dis)
        feature_dic['*curvature_dis_mean'] = cur_dis.mean()
        feature_dic['*curvature_dis_min'] = cur_dis.min()
        feature_dic['*curvature_dis_max'] = cur_dis.max()
        feature_dic['*curvature_dis_max-'] = cur_dis.max() - cur_dis.min()
        feature_dic['*curvature_dis_var'] = cur_dis.var()


def dotMinus(dot1, dot2):
    """
    dot2 - dot1 (list - list)
    """
    res = []
    for i in range(len(dot2)):
        res.append(dot2[i] - dot1[i])
    return res


def get_curvature_rate(feature_dic, dots):
    c_angle_degree = []
    l_dis = []
    for i in range(len(dots)):
        c_angle_degree.append(math.atan(dots[i][1] / (eps + dots[i][0])))
        l_dis.append(sp.spatial.distance.euclidean((dots[i][0], dots[i][1]), (0, 0)))

    c_cu = []
    if len(l_dis) > 1:
        for i in range(len(l_dis) - 1):
            c_cu.append((c_angle_degree[i + 1] - c_angle_degree[i]) / (eps + l_dis[i + 1] - l_dis[i]))

        c_cu = np.array(c_cu)

        feature_dic['*curvature_rate_med'] = np.median(c_cu)
        feature_dic['*curvature_rate_mean'] = c_cu.mean()
        feature_dic['*curvature_rate_min'] = c_cu.min()
        feature_dic['*curvature_rate_max'] = c_cu.max()
        feature_dic['*curvature_rate_max-'] = c_cu.max() - c_cu.min()
        feature_dic['*curvature_rate_var'] = c_cu.var()

    if len(c_cu) > 1:
        c_v = []
        for i in range(len(c_cu) - 1):
            c_v.append((c_cu[i + 1] - c_cu[i]) / (eps + l_dis[i + 1] - l_dis[i]))

        c_v = np.array(c_v)

        feature_dic['*curvature_rate_vmed'] = np.median(c_v)
        feature_dic['*curvature_rate_vmean'] = c_v.mean()
        feature_dic['*curvature_rate_vmin'] = c_v.min()
        feature_dic['*curvature_rate_vmax'] = c_v.max()
        feature_dic['*curvature_rate_vmax-'] = c_v.max() - c_v.min()
        feature_dic['*curvature_rate_vvar'] = c_v.var()


def get_horizon_angle(feature_dic, dots):
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
        h_angle_degree.append(math.atan(point_minus[i][1] / (eps + point_minus[i][0])))
    h_angle_degree = np.array(h_angle_degree)
    if len(h_angle_degree):
        feature_dic['*h_angle_mean'] = h_angle_degree.mean()
        feature_dic['*h_angle_min'] = h_angle_degree.min()
        feature_dic['*h_angle_max'] = h_angle_degree.max()
        feature_dic['*h_angle_max_min'] = h_angle_degree.max() - h_angle_degree.min()
        feature_dic['*h_angle_var'] = h_angle_degree.var()
        feature_dic['*kurt_angle'] = sp.stats.kurtosis(h_angle_degree)

    h_angle_speed = []
    for i in range(len(h_angle_degree) - 1):
        h_angle_speed.append((h_angle_degree[i + 1] - h_angle_degree[i]) / (eps + (dots[i + 1][2] - dots[i][2])))

    h_angle_speed = np.array(h_angle_speed)
    if len(h_angle_speed):
        feature_dic['*h_angle_speed_mean'] = h_angle_speed.mean()
        feature_dic['*h_angle_speed_min'] = h_angle_speed.min()
        feature_dic['*h_angle_speed_max'] = h_angle_speed.max()
        feature_dic['*h_angle_speed_var'] = h_angle_speed.var()

    h_angle_acc = []
    for i in range(len(h_angle_speed) - 1):
        h_angle_acc.append((h_angle_speed[i + 1] - h_angle_speed[i]) / (eps + (dots[i + 1][2] - dots[i][2])))

    h_angle_acc = np.array(h_angle_acc)
    if len(h_angle_acc):
        feature_dic['*h_angle_acc_mean'] = h_angle_acc.mean()
        feature_dic['*h_angle_acc_min'] = h_angle_acc.min()
        feature_dic['*h_angle_acc_max'] = h_angle_acc.max()
        feature_dic['*h_angle_acc_var'] = h_angle_acc.var()
        feature_dic['*h_angle_acc_0'] = h_angle_acc[0]
        feature_dic['*h_angle_acc_-1'] = h_angle_acc[-1]


def dotProduct(dot1, dot2):
    """
    dot1 . dot2
    """
    res = 0
    for i in range(len(dot2)):
        res += (dot1[i] * dot2[i])
    return res


def toward_dest(feature_dic, dots, dest_point, x_only=False, y_only=False):
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
    if len(tmp):
        # add the feture to the dictionary
        feature_dic['*toward_dest_med'] = np.median(tmp)
        feature_dic['*toward_dest_mean'] = tmp.mean()
        feature_dic['*toward_dest_min'] = tmp.min()
        feature_dic['*toward_dest_max'] = tmp.max()
        feature_dic['*toward_dest_var'] = tmp.var()
        feature_dic['*toward_dist_dis'] = tmp.max() - tmp.min()

    tmp = []
    if len(dist_to_dest) == 1:
        tmp.append(0)
    else:
        for i in range(len(dist_to_dest) - 1):
            tmp.append(dotProduct(dist_to_dest[i], dist_to_dest[i + 1]) / (eps +
                                                                           dist(dist_to_dest[i], [0, 0]) * dist(
                                                                               dist_to_dest[i + 1], [0, 0])))
    tmp = np.array(tmp)
    if len(tmp):
        feature_dic['*aim_dot_product_mean'] = tmp.mean()
        feature_dic['*aim_dist_dot_product_min'] = tmp.min()
        feature_dic['*aim_dot_product_max'] = tmp.max()
        feature_dic['*aim_dot_product_var'] = tmp.var()
        feature_dic['*aim_kurt_angle'] = sp.stats.kurtosis(tmp)


def get_angle_change(feature_dic, dots):
    """
    get the angle changes along the path
    A, B, C
    angle between BA, BC
    """
    point_minus = []
    for i in range(len(dots) - 1):
        point_minus.append(dotMinus(dots[i + 1], dots[i]))

    tmp = []
    if len(point_minus) == 1:
        tmp.append(0)
    else:
        for i in range(len(point_minus) - 1):
            tmp.append(dotProduct(point_minus[i], point_minus[i + 1]) / (eps +
                                                                         dist(point_minus[i], [0, 0]) * dist(
                                                                             point_minus[i + 1], [0, 0])))
    tmp = np.array(tmp)
    # add the feture to the dictionary
    if len(tmp):
        feature_dic['*angle_mean'] = tmp.mean()
        feature_dic['*angle_min'] = tmp.min()
        feature_dic['*angle_max'] = tmp.max()
        feature_dic['*angle_var'] = tmp.var()
        feature_dic['*angle_kurt'] = sp.stats.kurtosis(tmp)

    angle_v = []
    # get angle velocity
    angle_change = list(tmp)
    for i in range(len(angle_change) - 1):
        angle_v.append((angle_change[i + 1] - angle_change[i]) / (eps + (dots[i + 1][2] - dots[i][2])))
    angle_v = np.array(angle_v)
    if len(angle_v) > 0:
        feature_dic['*angle_v_mean'] = angle_v.mean()
        feature_dic['*angle_v_min'] = angle_v.min()
        feature_dic['*angle_v_max'] = angle_v.max()
        feature_dic['*angle_v_var'] = angle_v.var()
        feature_dic['*angle_v_kurt'] = sp.stats.kurtosis(angle_v)

    # get angle acc
    angle_acc = []
    angle_v = list(angle_v)
    if len(angle_v) > 1:
        for i in range(len(angle_v) - 1):
            angle_acc.append((angle_v[i + 1] - angle_v[i]) / (eps + (dots[i + 1][2] - dots[i][2])))
        angle_acc = np.array(angle_acc)
        feature_dic['*angle_ac_mean'] = angle_acc.mean()
        feature_dic['*angle_acc_min'] = angle_acc.min()
        feature_dic['*angle_acc_max'] = angle_acc.max()
        feature_dic['*angle_acc_ji'] = angle_acc.max() - angle_acc.min()
        feature_dic['*angle_acc_var'] = angle_acc.var()
        feature_dic['*angle_acc_0'] = angle_acc[0]
        feature_dic['*angle_acc_-1'] = angle_acc[-1]


def get_length_dots(feature_dic, dots):
    """
    1. the number of points of the dots
    2. the length of the dots
    3. cai rate
    """

    def pow_2(x):
        return x * x

    sum = 0

    all_p = set()
    for i in range(len(dots)):
        all_p.add((dots[i][0], dots[i][1]))
    feature_dic['cai_point'] = len(all_p) / float(len(dots))
    for i in range(len(dots) - 1):
        sum += math.sqrt(pow_2(dots[i + 1][1] - dots[i][1]) + pow_2(dots[i + 1][0] - dots[i][0]))
    feature_dic['*dots_num'] = len(dots)
    feature_dic['*dots_length'] = sum


def get_time_feature(feature_dic, dots):
    """
    get features related to time series(time only)
    """
    time_list = []
    for i in range(len(dots)):
        time_list.append(dots[i][2])
    time_list = np.array(time_list)

    feature_dic['*time_duration'] = time_list.max() - time_list.min()


def get_smooth(feature_dic, dots, axis):
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
    feature_dic['*smooth_1_' + str(axis)] = smooth_1


def get_feature(df):
    """
    df is one line of the points
    """
    points = []
    dots = []

    for point in df.trajectory[:-1].split(';'):
        point = point.split(',')
        dots.append((float(point[0]), float(point[1]), float(point[2])))
        points.append(((float(point[0]), float(point[1])), float(point[2])))

    xs = pd.Series([point[0][0] for point in points])

    ys = pd.Series([point[0][1] for point in points])

    aim = df.aim.split(',')
    aim = (float(aim[0]), float(aim[1]))

    distance_deltas = pd.Series(
        [sp.spatial.distance.euclidean(points[i][0], points[i + 1][0]) for i in range(len(points) - 1)])

    time_deltas = pd.Series([points[i + 1][1] - points[i][1] for i in range(len(points) - 1)])
    xs_deltas = xs.diff(1)
    ys_deltas = ys.diff(1)

    speeds = pd.Series(
        [np.log1p(distance) - np.log1p(delta) for (distance, delta) in zip(distance_deltas, time_deltas)])
    angles = pd.Series(
        [np.log1p((points[i + 1][0][1] - points[i][0][1])) - np.log1p((points[i + 1][0][0] - points[i][0][0])) for i in
         range(len(points) - 1)])

    speed_diff = speeds.diff(1).dropna()
    angle_diff = angles.diff(1).dropna()

    distance_aim_deltas = pd.Series([sp.spatial.distance.euclidean(points[i][0], aim) for i in range(len(points))])
    distance_aim_deltas_diff = distance_aim_deltas.diff(1).dropna()

    # get the difference of the speed and angle
    df['speed_diff_median'] = speed_diff.median()
    df['speed_diff_mean'] = speed_diff.mean()
    df['speed_diff_var'] = speed_diff.var()
    df['speed_diff_max'] = speed_diff.max()
    df['angle_diff_var'] = angle_diff.var()

    # get the difference of the time
    df['time_delta_min'] = time_deltas.min()
    df['time_delta_max'] = time_deltas.max()
    df['time_delta_var'] = time_deltas.var()

    # distance
    df['distance_deltas_max'] = distance_deltas.max()
    df['distance_deltas_0_count'] = len(distance_deltas[distance_deltas == 0])

    # toward the dest
    df['aim_distance_last'] = distance_aim_deltas.values[-1]
    df['aim_distance_diff_max'] = distance_aim_deltas_diff.max()
    df['aim_distance_diff_var'] = distance_aim_deltas_diff.var()

    df['mean_speed'] = speeds.mean()
    df['median_speed'] = speeds.median()
    df['var_speed'] = speeds.var()

    df['max_angle'] = angles.max()
    df['var_angle'] = angles.var()
    df['kurt_angle'] = angles.kurt()  # the peak

    df['y_min'] = ys.min()
    df['y_max'] = ys.max()
    df['y_var'] = ys.var()

    df['x_min'] = xs.min()
    df['x_max'] = xs.max()
    df['x_var'] = xs.var()

    # initial values
    df['x_init'] = xs.values[0]
    df['y_init'] = ys.values[0]

    df['x_back_num'] = min((xs.diff(1).dropna() > 0).sum(), (xs.diff(1).dropna() < 0).sum())
    df['y_back_num'] = min((ys.diff(1).dropna() > 0).sum(), (ys.diff(1).dropna() < 0).sum())

    df['xs_delta_var'] = xs_deltas.var()
    df['xs_delta_max'] = xs_deltas.max()
    df['xs_delta_min'] = xs_deltas.min()
    df['time_deltas_0_count'] = len(time_deltas[time_deltas == 0])

    get_single_feature(df)

    # add our features
    # point coordinate features
    get_dis(df, dots, x_only=True)
    get_dis(df, dots, y_only=True)

    # velocity features
    get_velocity(df, dots)
    get_velocity(df, dots, x_only=True)
    get_velocity(df, dots, y_only=True)

    # acc features
    get_acc_speed(df, dots, y_only=True)
    get_acc_speed(df, dots, x_only=True)
    get_acc_speed(df, dots)

    # get density of x, y, t
    for i in range(3):
        get_density(df, dots, i)

    # get curvature_distance in the trace #4
    curvature_distance(df, dots)

    # toward destination
    toward_dest(df, dots, aim)

    # angle change between two lines
    get_angle_change(df, dots)
    get_horizon_angle(df, dots)

    # lenght of dots
    get_length_dots(df, dots)

    # time length
    get_time_feature(df, dots)

    # smooth of the curve
    for i in range(2):
        get_smooth(df, dots, i)

    get_curvature_rate(df, dots)
    return df.to_frame().T


def get_single_feature(df):
    points = []

    for point in df.trajectory[:-1].split(';'):
        point = point.split(',')
        points.append(((float(point[0]), float(point[1])), float(point[2])))

    aim = df.aim.split(',')
    aim = (float(aim[0]), float(aim[1]))

    aim_angle = pd.Series([np.log1p(point[0][1] - aim[1]) - np.log1p(point[0][0] - aim[0]) for point in points])
    aim_angle_diff = aim_angle.diff(1).dropna()

    df['aim_angle_last'] = aim_angle.values[-1]
    df['aim_angle_diff_max'] = aim_angle_diff.max()
    df['aim_angle_diff_var'] = aim_angle_diff.var()

    if len(aim_angle_diff) > 0:
        df['aim_angle_diff_last'] = aim_angle_diff.values[-1]
    else:
        df['*aim_angle_diff_last'] = -1
    return df.to_frame().T


def make_train_set(reuse=False):
    dump_path = os.path.join(cache, 'train.hdf')
    if not reuse and os.path.exists(dump_path):
        os.remove(dump_path)
    if os.path.exists(dump_path):
        train = pd.read_hdf(dump_path, 'all')
    else:
        train = pd.read_csv(train_path, sep=' ', header=None, names=['id', 'trajectory', 'aim', 'label'])
        train['count'] = train.trajectory.map(lambda x: len(x.split(';')))
        train = applyParallel(train.iterrows(), get_feature).sort_values(by='id')
        train.to_hdf(dump_path, 'all')
    return train


def make_test_set(reuse=False):
    dump_path = os.path.join(cache, 'test.hdf')
    if not reuse and os.path.exists(dump_path):
        os.remove(dump_path)
    if os.path.exists(dump_path):
        test = pd.read_hdf(dump_path, 'all')
    else:
        test = pd.read_csv(test_path, sep=' ', header=None, names=['id', 'trajectory', 'aim'])
        test['count'] = test.trajectory.map(lambda x: len(x.split(';')))
        test = applyParallel(test.iterrows(), get_feature).sort_values(by='id')
        test.to_hdf(dump_path, 'all')
    return test


def get_test_error(model_file, test_data, labels, threshold=0.5):
    gbm = lgb.Booster(model_file=model_file)
    pred = gbm.predict(test_data)

    corr = (pred > threshold) == labels
    acc = sum(corr) / len(labels)
    pre_neg_sum = len(labels) - sum(labels)
    true_neg_sum = len(pred) - sum(pred > threshold)
    neg_pos = sum(np.logical_and(corr, np.logical_not(labels)))
    precision = neg_pos / pre_neg_sum
    recall = neg_pos / true_neg_sum
    print('Acc:%s\tPrecision:%s\tRecall:%s' % (acc, precision, recall))


def prepare_data(reuse_train=False, reuse_test=False):
    draw_if = False
    train, test = make_train_set(reuse=reuse_train), make_test_set(reuse=reuse_test)
    if draw_if:
        train.reset_index().rename(columns={'index': 'idx'}).apply(draw, axis=1)
    training_data, label = train.drop(['id', 'trajectory', 'aim', 'label'], axis=1).astype(float), train['label']
    sub_training_data, instanceIDs = test.drop(['id', 'trajectory', 'aim'], axis=1).astype(float), test['id']
    print (training_data.shape)

    return training_data, label, sub_training_data, instanceIDs


def lgb_train(train_x, train_y, test_x, test_y, save_model_file):
    # lgb
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(test_x, test_y, reference=lgb_train)

    print (train_x.shape)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'num_leaves': 7,
        'learning_rate': 0.05,
        'feature_fraction': 0.83,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': 0
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=280,
                    valid_sets=[lgb_train, lgb_eval],
                    verbose_eval=True)
    gbm.save_model(save_model_file)
    get_test_error(save_model_file, test_x, test_y)


def inference(model_file, vali_data, instanceIDs):
    gbm = lgb.Booster(model_file=model_file)
    y = gbm.predict(vali_data)
    res = instanceIDs.to_frame()
    res['prob'] = y
    res['id'] = res['id'].astype(int)
    res = res.sort_values(by='prob')
    res.iloc[0:20000].id.to_csv(os.path.join(sub, 'BDC20160706.txt'), header=None, index=False)


def get_data(model_file, vali_data, instanceIDs):
    gbm = lgb.Booster(model_file=model_file)
    y = gbm.predict(vali_data)
    res = instanceIDs.to_frame()
    res['prob'] = y
    res['id'] = res['id'].astype(int)
    res = res.sort_values(by='prob')
    ans = np.array(res.iloc[-50000:].id).astype(int)
    with open('./data/zheng2.txt', 'w') as f:
        print(len(ans))
        for i in range(len(ans)):
            f.write(str(ans[i]) + '\n')


if __name__ == '__main__':
    training_data, label, sub_training_data, instanceIDs = prepare_data(reuse_train=False, reuse_test=False)
    # train_x, test_x, train_y, test_y = train_test_split(training_data, label, test_size=0.01, random_state=0)

    # lgb_train(train_x, train_y, test_x, test_y, './lgb_model')

    # inference('./lgb_model', sub_training_data, instanceIDs)
    get_data('./lgb_model', sub_training_data, instanceIDs)
