import numpy as np
from util import loadData
from sklearn.model_selection import train_test_split
from model import naiveBayes,decisionTree,knn,randomForest,svm
from sklearn.preprocessing import StandardScaler
from FeatureSelection import PreProcess
from DataBalance import SMOTE


def generateDataset(length, X, Y):
    DiX = []
    DiY = []
    for _ in range(length):
        p = np.random.randint(1, length)
        DiX.append(X[p])
        DiY.append(Y[p])
    return np.array(DiX),np.array(DiY)

def bagging(n_classes,functions,k=5,mode1=False,mode2=True):

    '''
    :param functions:
    :param k:
    :param mode1: mode1==True，即使用SMOTE数据平衡处理
    :param mode2: mode2==True，即使用OOM数据集训练模型，mode2 == False，使用SCM数据集训练模型
    :return:
    '''

    for p in range(len(functions)):
        #k = 5
        count = 0
        _sum = 0
        acc = {}
        pred = []

        #交叉验证（感觉有些数据集没必要，数据量很大）
        for i in range(k):
            # dix = []
            # diy = []
            # dix, diy = generateDataset(len(X), X, y)
            # acc[i],b,y_score = functions[p](dix, X,diy, y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                                random_state=0)

            '''SMOTE处理数据不平衡'''
            if(mode1):
                X_train, y_train = SMOTE.SMOTE(X_train, y_train)

            functions[p](n_classes,X_train, X_test, y_train, y_test,mode2)

if __name__ == '__main__':

    '''使用OOM数据集'''
    # filePath = "../Dataset/OOM/camel-1.6.csv"
    '''使用SCM数据集'''
    filePath = "../Dataset/SCM/JM1.csv"
    X,y = loadData.getData(filePath)

    sc = StandardScaler()
    X = sc.fit_transform(X)

    #使用特征选择工具
    X,y = PreProcess.featureSelection(X,y,mode=1)

    # 对标签进行one-hot编码
    X, y, n_classes = PreProcess.labelBinarize(X, y)

    functions = [randomForest.RandomForest, knn.Knn, decisionTree.DecisionTree, naiveBayes.NaiveBayes, svm.Svm]

    bagging(n_classes,functions,k=5,mode1=False,mode2=False)






