import numpy as np
import math
from random import shuffle
import functions
import pandas as pd
import random

GlobalBestValue1 = 0
Sn = 50
D =30
FuncValue = np.zeros([Sn, 1])
Fitness = np.zeros([Sn, 1])
Iter = 1500
Limit = 100
BestSol = np.zeros([D, ])
Probability = np.zeros([Sn, 1])
Trail = np.zeros([Sn, 1])
Fes = 0
GlobalBestValue = 0
ite = 0
k1 = random.randint(1, 49)
k2 = 20
func_num = 8
f_num = 28
T= 1
MaxFes = 10000 *D
arr = np.zeros([T, 1])
#arr = np.array(T)
#cec_functions = functions.CEC_functions(D)
a = 0
b = 0
ites = 0
it = np.zeros([16000, 1])


def T1(x):
    result = 0.0
    for i in range(D):
        result += 1/6*math.pow(x[i],6) - 52/25*(math.pow(x[i],5)) + 39/80*(math.pow(x[i],4)) + 71/10*(math.pow(x[i],3)) - 79/20*(math.pow(x[i],2)) - x[i] + 1/10
    return result

def T2(x):
    result = 0.0
    a = [-500, 2.5, 1.666666666, 1.25, 1.0, 0.8333333, 0.714285714, 0.625,0.555555555, 1.0, -43.6363636, 0.41666666, 0.384615384,
         0.357142857,0.3333333,0.3125,0.294117647,0.277777777,0.263157894,0.25,0.238095238,0.227272727,0.217391304,0.208333333,0.2,
         0.192307692,0.185185185,0.178571428,0.344827586,0.6666666,-15.48387097,0.15625,0.1515151,0.14705882,0.14285712,0.138888888,
         0.135135135,0.131578947,0.128205128,0.125,0.121951219,0.119047619,0.116279069,0.113636363,0.1111111,0.108695652,0.106382978,0.208333333,0.408163265,0.8]
    for i in range(50):
        result += a[i] * math.pow(x,i)
    return result

def T3(x):
    result = 0.0
    for i in range(D):
        result += 0.000089248 * x[i] - 0.0218343 * (math.pow(x[i], 2)) + \
                  0.998266 * (math.pow(x[i], 3)) - 1.6995 * (math.pow(x[i], 4)) + 0.2 * (math.pow(x[i], 5))
    return result

def T4(x):
    result = 0.0
    for i in range(D):
        result += 4*math.pow(x[i], 2) - 4 * (math.pow(x[i], 3)) + (math.pow(x[i], 4))
    return result

def T5(x):
    result = 0.0
    result = 2*math.pow(x[D-2], 2) - 1.05 * (math.pow(x[D-2], 4))\
             + 1/6 * (math.pow(x[D-2], 6)) - x[D-2] * x[D-1] + math.pow(x[D-1],2)
    return result

def T6(x):
    result = 0.0
    for i in range(D):
        result += math.pow(x[i], 6) - 15 * (math.pow(x[i], 4)) + 27 * (math.pow(x[i], 2)) + 250
    return result

def T7(x):
    result = 0.0
    for i in range(D):
        result += math.pow(x[i], 4) - 3 * (math.pow(x[i], 3)) + 1.5 * (math.pow(x[i], 2)) + 10*x[i]
    return result

def F1(x):
    result = 0.0
    for i in range(D):
        result += x[i]**2
    return result

# 测试函数F2  Elliptic [-100,100]
def F2(x):
    result = 0.0
    for i in range(D):
        result += math.pow(10.0, 6.0*i/(D-1))*x[i]*x[i]
    return result

# 测试函数 F3 SumSquare [-10,10]
def F3(x):
    result = 0.0
    for i in range(D):
        result += (i+1)*x[i]*x[i]
    return result

# 测试函数 F4 SumPower [-1,1]
def F4(x):
    result = 0.0
    for i in range(D):
        result += math.pow(math.fabs(x[i]), i+2)
    return result

# 测试函数F5 schwefel 2.22 [-10,10]
def F5(x):
    result = 0
    tmp1 = 0
    tmp2 = 1.0
    for i in range(D):
        tmp = abs(x[i])
        tmp1 += tmp
        tmp2 *= tmp
    result = tmp1 +tmp2
    return result

# 测试函数F6 schwefel 2.21 [-100,100]
def F6(x):
    result = np.max(np.abs(x))
    return result

# 测试函数F7 Step [-100,100]
def F7(x):
    result = 0
    for i in range(D):
        result += (math.floor(x[i]+0.5))*(math.floor(x[i]+0.5))
    return result

# 测试函数F8 Exponential [-10,10]
def F8(x):
    result = np.exp(0.5*np.sum(x))
    return result

# 测试函数F9 Quartic [-1.28,1.28]
def F9(x):
    result = 0.0
    for i in range(D):
        result += (i+1.0)*x[i]*x[i]*x[i]*x[i]
    result += np.random.rand()
    return result

# 测试函数F10 Rosenbrock [-5,10]
def F10(x):
    result = 0
    tmp1 = 0
    tmp2 = 0
    for i in range(D-1):
        tmp1 = 100*(x[i]*x[i]-x[i+1])*(x[i]*x[i]-x[i+1])
        tmp2 = (x[i]-1)*(x[i]-1)
        result += tmp1 + tmp2
    return result

# 测试函数F11 Rastrigrin [-5.12,5.12]
def F11(x):
    result = 0
    for i in range(D):
        result += (x[i] * x[i] - 10 * math.cos(2 * math.pi * x[i]) + 10)
    return result

# 测试函数F12 NCRastrigin [-5.12,5.12]
def F12(x):
    result = 0.0
    for i in range(D):
        if np.abs(x[i]) < 0.5:
            result += (x[i] * x[i] - 10 * math.cos(2 * math.pi * x[i]) + 10)
        else:
            y = round(2*x[i])/2
            result += (y * y - 10 * math.cos(2 * math.pi * y) + 10)
    return result

# 测试函数F13 Griewank [-10000,10000]
def F13(x):
    result = 0
    tmp1 = 0.0
    tmp2 = 1.0
    for i in range(D):
        tmp1 += (x[i]**2)/4000
        tmp2 *= math.cos(x[i]/math.sqrt(i+1.0))
    result = tmp1 - tmp2 + 1
    return result

# 测试函数F14 Schwefel2.26 [-500,500]
def F14(x):
    result = 0.0
    for i in range(D):
        result += (x[i]) * math.sin(math.sqrt(math.fabs(x[i])))
    result = 418.98288727243380 * D - result
    return result

# 测试函数F15 Ackley [-50,50]
def F15(x):
    result = 0
    tmp1 = 0.0
    tmp2 = 0.0
    for i in range(D):
        tmp1 += x[i]**2
        tmp2 += math.cos(2*math.pi*x[i])
    tmp1 /= D
    tmp1 = -0.2*math.sqrt(tmp1)
    tmp1 = -20*math.exp(tmp1)
    tmp2 /= D
    tmp2 = math.exp(tmp2)
    result = tmp1 - tmp2 + 20 + math.exp(1.0)
    return result

# 测试函数F16 Penalized1 [-100,100]
def F16(x):
    result1 = 0.0
    result2 = 0.0
    for i in range(D-1):
        y1 = 1 + (x[i]+1)/4
        y2 = 1 + (x[i+1]+1)/4
        if x[i] > 10:
            u = 100 * np.power((x[i] - 10), 4)
        elif x[i] < -10:
            u = 100 * np.power((-x[i] - 10), 4)
        else:
            u = 0
        result1 += (np.square(y1 - 1)*(1 + 10 * np.square(np.sin(math.pi*y2))))
        result2 += u
    if x[D-1] > 10:
        u = 100 * np.power((x[D-1] - 10), 4)
    elif x[D-1] < -10:
        u = 100 * np.power((-x[D-1] - 10), 4)
    else:
        u = 0
    result2 += u
    y1 = 1 + (x[0]+1)/4
    y2 = 1 + (x[D-1]+1)/4
    result = math.pi/D * (10 * np.square(np.sin(math.pi * y1)) + result1 + np.square(y2 - 1)) + result2
    return result

# 测试函数F17 Penalized2 [-100,100]
def F17(x):
    result = 0
    result1 = 0
    result2 = 0
    result3 = 0
    for i in range(D):
        if x[i] > 5:
            u = 100*math.pow(x[i]-5, 4)
        elif x[i] < -5:
            u = 100*math.pow((-x[i]-5), 4)
        else:
            u = 0
            result1 = result1 + u
    for i in range(D-1):
        result2 = result2 + (x[i]-1)*(x[i]-1)*(1+math.sin(3*math.pi*x[i+1])*math.sin(3*math.pi*x[i+1]))
    result3 = result2 + math.sin(3*math.pi*x[0])*math.sin(3*math.pi*x[0])+(x[D-1]-1)*(x[D-1]-1)*(1+math.sin(2*math.pi*x[D-1])*math.sin(2*math.pi*x[D-1]))
    result = 0.1*result3 + result1
    return result

# 测试函数F18 Alpine [-10,10]
def F18(x):
    result = 0.0
    for i in range(D-1):
        result += np.abs(x[i]*np.sin(x[i]) + 0.1*x[i])
    return result

# 测试函数F19 Levy [-10,10]
def F19(x):
    result = 0
    result1 = 0
    result2 = 0
    for i in range(D - 1):
        result2 += (x[i] - 1) * (x[i] - 1) * (1 + math.sin(3 * math.pi * x[i + 1]) * math.sin(3 * math.pi * x[i + 1]))
        result1 = result2 + math.sin(3 * math.pi * x[1]) * math.sin(3 * math.pi * x[1])
        result = result1 + abs(x[D - 1] - 1) * (1 + math.sin(3 * math.pi * x[D - 1]) * math.sin(3 * math.pi * x[D - 1]))
    return result

# 测试函数F20 Weierstrass [-1,1]
def F20(x):
    result1 = 0.0
    result2 = 0.0
    for i in range(D):
        for k in range(21):
            result1 += np.power(0.5, k)*np.cos(2*math.pi*np.power(3, k)*(x[i] + 0.5))
    for k in range(21):
        result2 += np.power(0.5, k)*np.cos(2*math.pi*np.power(3, k)*0.5)
    result = result1 - D*result2
    return result

# 测试函数F21 Himmelblau [-5,5]
def F21(x):
    result = 0
    for i in range(D):
        result += (math.pow(x[i], 4) - 16 * x[i] * x[i] + 5 * x[i]) / D
    return result

# 测试函数F22 Michalewicz [0,math.pi]
def F22(x):
    result = 0
    result1 = 0
    for i in range(D):
        result1 = math.sin((i + 1) * x[i] * x[i] / math.pi)
        result += math.sin(x[i]) * math.pow(result1, 20)
    return -result

func_list = [T1, T2, T3, T4, T5, T6, T7, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21, F22]

bound = np.array([
    [-2,11],
    [1,2],
    [0,10],
    [-5,5],
    [-5,5],
    [-5,5],
    [-5,5],
    [-100, 100],
    [-100, 100],
    [-10, 10],
    [-1, 1],
    [-10, 10],
    [-100, 100],
    [-100, 100],
    [-10, 10],
    [-1.28, 1.28],
    [-5, 10],
    [-5.12, 5.12],
    [-5.12, 5.12],
    [-600, 600],
    [-500, 500],
    [-50, 50],
    [-100, 100],
    [-100, 100],
    [-10, 10],
    [-10, 10],
    [-1, 1],
    [-5, 5],
    [0, math.pi]
])

lb = bound[func_num - 1][0]
ub = bound[func_num - 1][1]

def function_value(x):
    global Fes
    Fes += 1
    result = func_list[func_num - 1](x)
    return result

def fitness(value):
    if value >= 0:
        return 1 / (1 + value)
    else:
        return 1 + abs(value)

def Distance(index):   
    distance = np.sqrt(np.sum(np.square(NectarSource[i] - NectarSource),axis = 1))
    meandistance = (np.sum(distance) / (Sn - 1)) * 1.0
    a = np.where(distance <= meandistance)[0]
    b = np.where(FuncValue == np.min(FuncValue[a]))[0][0]
    return b

def dim_perturbation(rnum):
    randnum = np.random.randint(2, rnum + 1)
    l = random.sample(range(0, D), randnum)
    theta = 2 * np.random.rand(randnum) - 1
    fai = 2 * np.random.rand(randnum)  * (3/4)
    '''else:
        randnum = np.random.randint(0, D)
        l = randnum
        theta = 2 * np.random.rand() - 1
        fai =  np.random.rand()  * (3/2)'''
    return theta, l, fai

def neighbor_list(i):
    if (i - k2) < 0:
        li = [i for i in range(Sn-k2+i, Sn)]
        li.extend([i for i in range(i+1)])
    else:
        li = [i for i in range(i-k2, i+1)]
    if (i + k1) >= Sn:
        li.extend([i for i in range(i+1, Sn)])
        li.extend([i for i in range(0, k2-Sn+i+1)])
    else:
        li.extend([i for i in range(i+1, i+k2+1)])

    li = np.where(FuncValue == np.min(FuncValue[li]))[0][0]
    return li

def cal_best(im):
    li = [x for x in range(Sn)]
    shuffle(li)
    if im in li[:k1 + 1]:
        li = li[:k1 + 1]
    else:
        li = li[:2*k1]
    '''if i in li[:2*k1+1]:
        li = li[:2*k1+1]
    else:
        li = li[:2*k1]'''
    bestindex = np.where(FuncValue == np.min(FuncValue[li]))[0][0]
    return bestindex

'''def rand_best(i):  
    li = random.randint(2, 49)
    bestindex = np.where(FuncValue == np.min(FuncValue[li]))[0][0]
    return bestindex'''

if __name__ == "__main__":
    for t in range(T):
        NectarSource = (ub - lb) * np.random.rand(Sn, D) + lb
        for i in range(Sn):
            FuncValue[i] = function_value(NectarSource[i])
            #FuncValue[i] = cec_functions.Y(NectarSource[i], f_num)
            Fitness[i] = fitness(FuncValue[i])
            # FuncValue[i] = cec_functions.Y(NectarSource[i],f_num)
        GlobalBestValue = np.min(FuncValue)
        BestSol = NectarSource[np.where(FuncValue == np.min(FuncValue))[0][0]]
        while Fes <= MaxFes:
            for i in range(Sn):
                best = cal_best(i)
                theta, rj, fai = dim_perturbation(D)
                while 1:
                    r1 = np.random.randint(0, Sn)
                    r2 = np.random.randint(0, Sn)
                    if (best != r1 and r1 != r2 and best != r2):
                        break
                j = np.random.randint(0, D)
                w = 1 - ite / Iter
                R = 2 * np.random.rand() - 1
                R1 = 1.5 * np.random.rand()
                V = np.copy(NectarSource[i])
                BestSol = BestSol.reshape(D, )
                V[j] = NectarSource[best][j] + R * (NectarSource[best][j] - NectarSource[r2][j])
                V[j] = np.where(V[j] > ub, ub, V[j])
                V[j] = np.where(V[j] < lb, lb, V[j])
                NewValue = function_value(V)
                #NewValue = cec_functions.Y(V, f_num)
                NewFitness = fitness(NewValue)
                if NewValue < FuncValue[i]:
                    # if NewFitness > Fitness[i]:
                    #NectarSource[i][rj] = np.copy(V[j])
                    NectarSource[i][j] = np.copy(V[j])
                    Trail[i] = 0
                    FuncValue[i] = np.copy(NewValue)
                    Fitness[i] = NewFitness
                    BestSol = NectarSource[np.where(FuncValue == np.min(FuncValue))[0][0]]
                else:
                    Trail[i] += 1
            Probability = Fitness / np.sum(Fitness)
            num = 0
            i = 0
            for i in range(Sn):
                best = neighbor_list(i)
                d = Distance(i)
                BestSol = BestSol.reshape(D, )
                while 1:
                    k = np.random.randint(0, Sn)
                    if (i != k and best != k):
                        break
                j = np.random.randint(0, D)
                R = 2 * np.random.rand() - 1
                R1 = 2 * np.random.rand() - 0.5
                V2 = np.copy(NectarSource[best])
                V2[j] = NectarSource[d][j] + R * (NectarSource[d][j] - NectarSource[k][j])
                V2[j] = np.where(V2[j] > ub, ub, V2[j])
                V2[j] = np.where(V2[j] < lb, lb, V2[j])
                NewValue2 = function_value(V2)
                #NewValue2 = cec_functions.Y(V2, f_num)
                Fes += 1
                if NewValue2 < FuncValue[best]:
                    NectarSource[best][j] = np.copy(V2[j])
                    Trail[best] = 0
                    FuncValue[best] = np.copy(NewValue2)
                    BestSol = NectarSource[np.where(FuncValue == np.min(FuncValue))[0][0]]
                if NewValue2 < FuncValue[best]:
                    NectarSource[best][j] = np.copy(V2[j])
                    Trail[best] = 0
                    FuncValue[best] = np.copy(NewValue2)
                    BestSol = NectarSource[np.where(FuncValue == np.min(FuncValue))[0][0]]
                else:
                    Trail[best] += 1
            for i in range(Sn):
                if Trail[i] >= Limit:
                    best = cal_best(i)
                    j = np.random.randint(0, D)
                    R = 2 * np.random.rand() - 1
                    while 1:
                        r1 = np.random.randint(0, Sn)
                        r2 = np.random.randint(0, Sn)
                        if (best != r1 and r1 != r2 and best != r2):
                            break
                    NectarSource[i] = NectarSource[best][j] + R * (NectarSource[r1][j] - NectarSource[r2][j])
                    NectarSource[i] = np.where(NectarSource[i] > ub, ub, NectarSource[i])
                    NectarSource[i] = np.where(NectarSource[i] < lb, lb, NectarSource[i])
                    Trail[i] = 0
                    FuncValue[i] = function_value(NectarSource[i])
                    #FuncValue[i] = cec_functions.Y(NectarSource[i], f_num)
                    #Fes += 1
                    Fitness[i] = fitness(FuncValue[i])
            if np.min(FuncValue) < GlobalBestValue:
                GlobalBestValue = np.min(FuncValue)
                BestSol = NectarSource[np.where(FuncValue == np.min(FuncValue))[0][0]]
            if (ite + 1) % 10 == 0:
                print(GlobalBestValue)
                it[ite] = GlobalBestValue
            ite += 1
        Fes = 0
        ite = 0
        dataframe = pd.DataFrame(it)
        print(GlobalBestValue)

