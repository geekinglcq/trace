## 特征 

### 坐标相关（8）  
+ x轴坐标值的均值方差和最值（4）  
+ y轴坐标值的均值方差和最值（4）   

### 速度相关（19）  
+ 移动速度的均值方差最值，零速度情况占比和得到的瞬时速度个数（6）  
+ 移动速度（仅考虑x轴）的均值方差最值，零速度情况占比和得到的瞬时速度个数（6）  
+ 移动速度（仅考虑y轴）的均值方差最值，零速度情况占比和得到的瞬时速度个数（6）   
+ 速度为0的时间占总时间的比重（1）

### 加速度相关（12）  
+ 移动加速度的均值方差最值（4）  
+ 移动加速度（仅考虑x轴）的均值方差最值（4）  
+ 移动加速度（仅考虑y轴）的均值方差最值（4）  

### 目标点相关（8）  
+ 过程点到目标点距离的均值方差最值（4）  
+ 过程点到目标点的向量中相邻向量的内积（4）  

### 角度相关（12）   
+ 移动轨迹角度变化的均值方差最值（4）  
+ 角速度(4)
+ 角加速度(4)

### 时间相关（7）
+ 时间均值，时间长度，方差（3）
+ 相邻时间的均值方差极值（4）

### 其他（2）  
+ 是否往回滑动   
+ x的点密度    

*特征说明详见pre.py的注释部分*

```
0th velocity_init_x_only_Ture_y_only_False: 数值大于60的都是正样本，有74个       figure_0      
3-th acc_max_x_only_False_y_only_True: 大于0.25的都是正样本， 455/2600，负样本最高0.21
5-th acc_var_x_only_False_y_only_False:大于2的都是正样本， 有470/2600 figure_2  
8-th velocity_max_x_only_False_y_only_False: 大于20的都是正样本，486/2600, figure_3  
10-th h_angle_acc_max: 大于0.02的都是正样本， 618/2600, 负样本最高0.011， figure_4  
11-th velocitu_min_x_only_False_y_only_False: 大于0.5的有84个，其中只有两个是正样本，其他82个都是负样本，且两个正样本看起来像噪音数据  figure_5    
13-th h_angle_speed_min: 小于0.2的全是正样本，431/2600，负样本最小-0.14  
20-th velocity_mean_x_only_False_y_only_True: 大于2.5的全是+，200/2600，-样本最高2.38  figure_6  
24-th go_back, 有返回的（值为1）全是正样本，1660/2600， figure_7  
35-th acc_min_x_only_False_y_only_False: 小于-2 的都是+ ，591/2600  39-th h_angle_acc_var: 大于1的都是+，279/2600，-样本最高7.4e-6.  
45-th velocity_max_x_only_True_y_only_False, 大于20的都是+，471/2600，-样本最高19.8，figure_9  
47-th acc_max_x_only_True_y_only_False, >0.5都是+，551/2600，负样本最高0.14, figure_10  
48-th acc_mean_x_only_False_y_only_False, <-1 都是+，464/4600，负样本最高-0.9， figure_11  
50-th h_angle_speed_mean, 所有负样本去枝都在-0.02~0.02之间，figure_12  
51-th h_angle_speed_var, >1的都是+，242/2600，负样本最高0.001  
58-th acc_max_x_only_False_y_only_False,635/2600, > 0.3都是正样本，负样本最高0.17，figure_13  

68-th velocity_min_x_only_False_y_only_True, 正样本全都是0， Figure_14    
69-th h_angle_speed_max, >0.3 全是正样本，346/2600，负样本最高0.12，Figure_15  
71-th *smooth_1_0, >1.5全是正样本，816/2600，负样本最高1.23，figure_16    
75-th acc_mean_x_only_True_y_only_False, < -1000都是正样本，451/2600， 负样本最小-0.9，Figure_17  
81-th velocity_mean_x_only_False_y_only_False, > 20 都是正样本，468/2600，负样本最高12，figure_18,   
91-th velocity_mean_x_only_True_y_only_False, >20都是正样本，455/2600，负样本最高12， figure_19  
94-th acc_min_x_only_False_y_only_True, <-1 都是正样本，430/2600， 负样本最小 -0.433

```