'''SMOTE处理数据不平衡'''
from collections import Counter
import numpy as np

def SMOTE(X,y):

    '''
     :param: X
     :param: y
     :param mode: int类型，如果mode = 0,则用通用的方式进行SMOTE过采样；如果mode = n, 表示对排名后n位的类别进行SMOTE过采样
    '''
    try:
        # 查看样本类别是否平衡
        temp = []
        for i in range(0,len(y)):
            temp.append(y[i,0])
        y_list = temp

        # print(Counter(y_list))
        # Counter({0: 900, 1: 100})


        # 使用imlbearn库中上采样方法中的SMOTE接口
        from imblearn.over_sampling import SMOTE
        # 定义SMOTE模型，random_state相当于随机数种子的作用
        smo = SMOTE(random_state=42,k_neighbors=5)
        X_smo, y_smo = smo.fit_resample(X, y)

        # print(Counter(y_smo))

        #将SMOTE生成的数据转换成模型指定输入类型
        y = []
        for i in range(0,len(y_smo)):
            temp = [y_smo[i]]
            y.append(temp)
        # print(y)
        y = np.array(y)
        X = X_smo

    except ValueError:
        from imblearn.over_sampling import RandomOverSampler
        randomSampler = RandomOverSampler(random_state=42)
        X_rand,y_rand = randomSampler.fit_resample(X,y)

        # y = []
        # for i in range(0, len(y_rand)):
        #     temp = [y_rand[i]]
        #     y.append(temp)
        # # print(y)
        y = np.array(y_rand)
        X = X_rand
    return X,y
