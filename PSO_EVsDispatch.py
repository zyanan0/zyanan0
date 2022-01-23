# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 11:32:28 2021
利用粒子群算法优化调度台区内EV,达到削峰填谷和调节峰谷差目的。 
@author: Air
"""
import time
time_start=time.time()
# In[] 导入数据

'''
导入用户负荷数据
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
load_user = pd.read_csv('FH.csv')
load_user = load_user[(load_user['rq']==20190801)&(load_user['FHDH']==-66)]
load_user.index = list(range(len(load_user)))
load_user = load_user.iloc[3000:3299,5:]#取300个用户
load_user.index = list(range(len(load_user)))
data = np.sum(load_user,axis=0)
data = pd.DataFrame(data)
data.index=pd.date_range('00:00:00',periods=len(data),freq='15min').time
data.columns = ['value']


# data.plot(figsize=(12,6)) 
# plt.rcParams['font.sans-serif']=['SimHei']#如果title是中文，matplotlib会乱码，这时需要加上下面这段代码
# plt.rcParams['axes.unicode_minus']=False#如果需要将数字设为负数，也可能出现乱码的情况，这时候可以加下面的代码
# plt.title('用户负荷数据')
# plt.grid()
# plt.show() 

'''
导入电动汽车数据
'''
#导入24点数据
load_EV0 = pd.read_csv('EVs.csv',header=None)
load_EV0 = load_EV0.T
load_EV0.index = pd.date_range('00:00:00',periods=96,freq='15min').time
load_EV0.columns = ['value']

#建立96点空集
zero = np.zeros((96,1))
load_EV = pd.DataFrame(zero,columns=['value'])
load_EV.index = pd.date_range('00:00:00',periods=len(load_EV),freq='15min').time
load_EV['value'] = load_EV.T

# 利用24点数据对根据索引名称对96点空集进行替换
load_EV['value'].fillna(load_EV0['value'],inplace=True)
load_EV = load_EV.fillna(method='ffill')

# load_EV.plot(figsize=(12,6))

# plt.title('电动汽车总负荷模拟数据')
# plt.grid()
# plt.show()


'''
绘制台区净负荷图像
'''

load_sum = load_EV + data

fig,ax = plt.subplots(figsize=(18,8))

# 绘制堆叠图（重点为bottom）
x = pd.date_range('00:00:00',periods=96,freq='15min') # 设置x轴
ax.bar(x, np.array(data['value']), width = 0.005,label="用户负荷")
ax.bar(x, np.array(load_EV['value']),width=0.005, alpha=0.8,  label="EVs",bottom=np.array(data['value']))
ax.plot(x,load_sum,linewidth=3,label="台区净负荷")

# 对于横轴为时间的可用下列语句处理其只显示时间（or日期'%Y-%m-%d'）
import matplotlib.dates as mdates    #處理日期
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S')) 

plt.title('Figure1')
plt.legend()
plt.grid()
plt.show()




# In[] 

'''
步骤1：初始化参数
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# PSO的参数
wmax = 2  # 惯性因子，一般取1
wmin = 1.8

c1max = 2.5  # 学习因子，一般取2
c1min = 1

c2max = 2.5
c2min = 1.5
  
r1 = None  # 为两个（0,1）之间的随机数
r2 = None
dim = 96  # 维度的维度,对应96个参数x1,...,x96
size = 100  # 种群大小，即种群中小鸟的个数
iter_num = 100  # 算法最大迭代次数
max_vel = 5  # 限制粒子的最大速度为0.5
fitneess_value_list = []  # 记录每次迭代过程中的种群适应度值变化


# In[]
'''
步骤2：这里定义一些参数，分别是计算适应度函数和计算约束惩罚项函数
'''

def cal_f(load_EV1):
    """计算粒子的的适应度值，也就是目标函数值，p 的维度是 size * 96 """
    listsum = (load_EV1+data['value'].values).tolist()
    sumv = np.array(min(listsum))
    sump = np.array(max(listsum))
    vp_rate = (sump-sumv)/sump #目标函数计算峰谷差率
    return vp_rate 

info0 = cal_f(load_sum['value'].values)    #测试示例

def calc_e1(load_EV1):
    """计算第一个约束的惩罚项"""
    delta = 0.5
    sum=load_EV.values.sum(axis=0)#0---纵向；1---横向
    e = abs(load_EV1.sum(axis=0)-sum)-delta
    if e<=50:
        return max(0, e/1000)
    else:
       if e<=100:
           return max(0, e/500)
       else:
           if e<=200:
                return max(0, e/100)
           else:
                return max(0, e/10)
           

def calc_Lj(e1):
    """根据每个粒子的约束惩罚项计算Lj权重值，e1, e2列向量，表示每个粒子的第1个第2个约束的惩罚项值"""
    # 注意防止分母为零的情况
    if e1.sum() <= 0:
        return 0
    else:
        L1 = e1.sum() / e1.sum()
    return L1   
 
    
# In[]
'''
步骤3：定义粒子群算法的速度更新函数，位置更新函数
'''

def velocity_update(V, X, pbest, gbest):
    """
    根据速度更新公式更新每个粒子的速度
     种群size=20
    :param V: 粒子当前的速度矩阵，100*96 的矩阵
    :param X: 粒子当前的位置矩阵，100*96 的矩阵
    :param pbest: 每个粒子历史最优位置，100*96 的矩阵
    :param gbest: 种群历史最优位置，1*96 的矩阵
    """
    r1 = np.random.random((size, 1))
    r2 = np.random.random((size, 1))
    V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)  # 直接对照公式写就好了
    # 防止越界处理
    V[V < -max_vel] = -max_vel
    V[V > max_vel] = max_vel
    return V


def position_update(X, V):
    """
    根据公式更新粒子的位置
    :param X: 粒子当前的位置矩阵，维度是 20*2
    :param V: 粒子当前的速度举着，维度是 20*2
    """
    X=X+V#更新位置
    size=np.shape(X)[0]#种群大小
    for i in range(size):#遍历每一个例子
        for j in range(95):
            if X[i][j]<=0 or X[i][j]>=40:#x的上下限约束
                X[i][j]=np.random.uniform(0,40,1)[0]#则在0到500随机生成一个数
    return X
    
# In[]
'''
步骤4：每个粒子历史最优位置更优函数，以及整个群体历史最优位置更新函数，和无约束约束优化代码类似，所不同的是添加了违反约束的处理过程
'''

def update_pbest(pbest, pbest_fitness, pbest_e, xi, xi_fitness, xi_e):
    """
    判断是否需要更新粒子的历史最优位置
    :param pbest: 历史最优位置
    :param pbest_fitness: 历史最优位置对应的适应度值
    :param pbest_e: 历史最优位置对应的约束惩罚项
    :param xi: 当前位置
    :param xi_fitness: 当前位置的适应度函数值
    :param xi_e: 当前位置的约束惩罚项
    :return:
    """
    # 下面的 0.0000001 是考虑到计算机的数值精度位置，值等同于0
    # 规则1，如果 pbest 和 xi 都没有违反约束，则取适应度小的
    if pbest_e <= 0.2 and xi_e <= 0.2:
        if pbest_fitness <= xi_fitness:
            return pbest, pbest_fitness, pbest_e
        else:
            return xi, xi_fitness, xi_e
    # 规则2，如果当前位置违反约束而历史最优没有违反约束，则取历史最优
    if pbest_e < 0.2 and xi_e >= 0.2:
        return pbest, pbest_fitness, pbest_e
    # 规则3，如果历史位置违反约束而当前位置没有违反约束，则取当前位置
    if pbest_e >= 0.2 and xi_e < 0.2:
        return xi, xi_fitness, xi_e
    # 规则4，如果两个都违反约束，则取适应度值小的
    if pbest_fitness <= xi_fitness:
        return pbest, pbest_fitness, pbest_e
    else:
        return xi, xi_fitness, xi_e

def update_gbest(gbest, gbest_fitness, gbest_e, pbest, pbest_fitness, pbest_e):
    """
    更新全局最优位置
    :param gbest: 上一次迭代的全局最优位置
    :param gbest_fitness: 上一次迭代的全局最优位置的适应度值
    :param gbest_e:上一次迭代的全局最优位置的约束惩罚项
    :param pbest:当前迭代种群的最优位置
    :param pbest_fitness:当前迭代的种群的最优位置的适应度值
    :param pbest_e:当前迭代的种群的最优位置的约束惩罚项
    :return:
    """
    # 先对种群，寻找约束惩罚项=0的最优个体，如果每个个体的约束惩罚项都大于0，就找适应度最小的个体
    pbest2 = np.concatenate([pbest, pbest_fitness.reshape(-1, 1), pbest_e.reshape(-1, 1)], axis=1)  # 将几个矩阵拼接成矩阵 ，4维矩阵（x,y,fitness,e）
    pbest2_1 = pbest2[pbest2[:, -1] <= 0.2]  # 找出没有违反约束的个体
    if len(pbest2_1) > 0:
        pbest2_1 = pbest2_1[pbest2_1[:, 96].argsort()]  # 根据适应度值排序
    else:
        pbest2_1 = pbest2[pbest2[:, 96].argsort()]  # 如果所有个体都违反约束，直接找出适应度值最小的
    # 当前迭代的最优个体
    pbesti, pbesti_fitness, pbesti_e = pbest2_1[0, :96], pbest2_1[0, 96], pbest2_1[0, 97]
    # 当前最优和全局最优比较
    # 如果两者都没有约束
    if gbest_e <= 0.2 and pbesti_e <= 0.2:
        if gbest_fitness < pbesti_fitness:
            return gbest, gbest_fitness, gbest_e
        else:
            return pbesti, pbesti_fitness, pbesti_e
    # 有一个违反约束而另一个没有违反约束
    if gbest_e <= 0.2 and pbesti_e > 0.2:
        return gbest, gbest_fitness, gbest_e
    if gbest_e > 0.2 and pbesti_e <= 0.2:
        return pbesti, pbesti_fitness, pbesti_e
    # 如果都违反约束，直接取适应度小的
    if gbest_fitness < pbesti_fitness:
        return gbest, gbest_fitness, gbest_e
    else:
        return pbesti, pbesti_fitness, pbesti_e    
    
    
# In[]

'''步骤5：PSO
# 初始化一个矩阵 info, 记录：
# 0、种群每个粒子的历史最优位置对应的适应度，
# 1、历史最优位置对应的惩罚项，
# 2、当前适应度，
# 3、当前目标函数值，
# 4、约束1惩罚项，
# 5、约束2惩罚项，
# 6、惩罚项的和
'''

# 所以列的维度是6
info = np.zeros((size, 6))

# 初始化种群的各个粒子的位置
# 用一个 200*96 的矩阵表示种群，每行表示一个粒子
# X = np.random.uniform(0, 40, size=(size, dim))
X = np.zeros((size,dim))
for n in range(size):
    X[n,:] = np.array(load_EV0.T)+np.random.uniform(-40, 40, size=(1, dim))
    
# 初始化种群的各个粒子的速度
V = np.random.uniform(-2, 2, size=(size, dim))

# 初始化粒子历史最优位置为当当前位置
pbest = X
# 计算每个粒子的适应度
for i in range(size):
    info[i, 3] = cal_f(X[i])  # 目标函数值
    info[i, 4] = calc_e1(X[i])  # 第一个约束的惩罚项
    # info[i, 5] = calc_e2(X[i])  # 第二个约束的惩罚项
    # 计算惩罚项的权重，及适应度值
L1 = calc_Lj(info[i, 4])
info[:, 2] = info[:, 3] + L1 * info[:, 4]   # 适应度值
info[:, 5] = L1 * info[:, 4]   # 惩罚项的加权求和

# 历史最优
info[:, 0] = info[:, 2]  # 粒子的历史最优位置对应的适应度值
info[:, 1] = info[:, 5]  # 粒子的历史最优位置对应的惩罚项值

# 全局最优
gbest_i = info[:, 0].argmin()  # 全局最优对应的粒子编号
gbest = X[gbest_i]  # 全局最优粒子的位置
gbest_fitness = info[gbest_i, 0]  # 全局最优位置对应的适应度值
gbest_e = info[gbest_i, 1]  # 全局最优位置对应的惩罚项

# 记录迭代过程的最优适应度值
fitneess_value_list.append(gbest_fitness)
# 接下来开始迭代
for j in range(iter_num):
    #更新权重
    w = wmax-(wmax-wmin)*j/iter_num
    # 更新学习因子
    c1 = c1max-(c1max-c1min)*j/iter_num
    c2 = c2min+(c2max-c2min)*j/iter_num
    
    # 更新速度
    V = velocity_update(V, X, pbest=pbest, gbest=gbest)
    # 更新位置
    X = position_update(X, V)
    # 计算每个粒子的目标函数和约束惩罚项
    for i in range(size):
        info[i, 3] = cal_f(X[i])  # 目标函数值
        info[i, 4] = calc_e1(X[i])  # 第一个约束的惩罚项
        # 计算惩罚项的权重，及适应度值
    L1 = calc_Lj(info[i, 4])
    info[:, 2] = info[:, 3] + L1 * info[:, 4] # 适应度值
    info[:, 5] = L1 * info[:, 4]  # 惩罚项的加权求和
    # 更新历史最优位置
    for i in range(size):
        pbesti = pbest[i]
        pbest_fitness = info[i, 0]
        pbest_e = info[i, 1]
        xi = X[i]
        xi_fitness = info[i, 2]
        xi_e = info[i, 5]
        # 计算更新个体历史最优
        pbesti, pbest_fitness, pbest_e = update_pbest(pbesti, pbest_fitness, pbest_e, xi, xi_fitness, xi_e)
        pbest[i] = pbesti
        info[i, 0] = pbest_fitness
        info[i, 1] = pbest_e
    # 更新全局最优
    pbest_fitness = info[:, 2]
    pbest_e = info[:, 5]
    gbest, gbest_fitness, gbest_e = update_gbest(gbest, gbest_fitness, gbest_e, pbest, pbest_fitness, pbest_e)
    # 记录当前迭代全局之硬度
    fitneess_value_list.append(gbest_fitness)

# 最后绘制适应度值曲线
load_EV_adj = gbest
print('迭代最优结果是：%.5f' % cal_f(gbest))
print('迭代最优变量是：', list(np.round(gbest,2)))
print('迭代约束惩罚项是：', gbest_e)

#迭代最优结果是：1.00491
#迭代最优变量是：x=1.00167, y=-0.00226
#迭代约束惩罚项是： 0.0
# 从结果看，有多个不同的解的目标函数值是相同的，多测试几次就发现了

# 绘图
plt.plot(fitneess_value_list, color='r')
plt.title('迭代过程')
plt.show()    


time_end=time.time()
print('totally cost',time_end-time_start)

# In[]

'''
绘制台区净负荷图像
'''

load_sum_adj = load_EV_adj + data.values.T
fig,ax = plt.subplots(figsize=(18,8))

# 绘制堆叠图（重点为bottom）
x = pd.date_range('00:00:00',periods=96,freq='15min') # 设置x轴
ax.bar(x, np.array(data['value']), width = 0.005,label="用户负荷")
# ax.bar(x, np.array(load_EV['value']),width=0.005, alpha=0.8,  label="EVs",bottom=np.array(data['value']))
ax.bar(x, load_EV_adj,width=0.005, alpha=0.8,  label="EVs",bottom=np.array(data['value']))
ax.plot(x,load_sum,linewidth=3,label="台区净负荷")
ax.plot(x,load_sum_adj.T,linewidth=3,label="调节后台区净负荷",color='r')
# 对于横轴为时间的可用下列语句处理其只显示时间（or日期'%Y-%m-%d'）
import matplotlib.dates as mdates    #處理日期
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S')) 

plt.title('Figure1')
plt.legend()
plt.grid()
plt.show()





