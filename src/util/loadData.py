from scipy.io import arff
import pandas as pd
import numpy as np

def getData(filePath):

    #数据读取
    df = pd.read_csv(filePath)

    #划分x，y
    columns = df.columns
    x = columns[:-1]
    y = columns[-1]
    x = df[x]
    y = df[[y]]

    #数据清洗

    # if(mode):
    #     '''修改标签：1 - True，0 - False'''
    #     column = columns[-1]
    #     # print(column)
    #     for i in range(0, len(df)):
    #         if (str(df.at[i,column]) == "b'N'"):
    #             df.at[i,column] = np.int64(1)
    #         else:
    #             df.at[i,column] = np.int64(0)
    # print(df)
    # df.to_csv("./Dataset/SCM/MC1.csv")

    '''1）去除掉不是数字的列'''
    for column in columns[:-1]:
        if(is_number(x.at[0,column]) == False):
            df = df.drop(column,axis=1)

    #重新获取x,y
    columns = df.columns
    x = columns[:-1]
    y = columns[-1]
    x = df[x]
    y = df[[y]]

    '''2）将字符串转数字'''
    for i in range(0,len(x)):
        for column in columns[:-1]:
            x.at[i,column] = float(x.at[i,column])

    print(df)

    # print(x,y)
    # print("type(x) = {},type(y) = {}".format(type(x.values), type(y.values)))
    print(x.values[0])
    return x.values,y.values


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

if __name__ == '__main__':
    filePath = "Dataset/SCM/MC1.csv"
    x,y = getData(filePath,mode=True)
    print("x = {},y = {}".format(x,y))