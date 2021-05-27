from src.FeatureSelection import CFS,PCA,IG
import numpy as np
from sklearn.preprocessing import label_binarize

def featureSelection(X,y,mode=1):
    '''
    :param mode: mode == 1,PCA; mode == 2,CFS; mode == 3,IG
    :return:
    '''

    if(mode == 1):
        X = PCA.pca(X,4)

    elif(mode == 2):
        cfs = CFS.CFS()
        cfs.fit(X[:, 1:-1], y)
        selected_cols = cfs.important_features

        X_x = []
        for j in range(0, len(X)):
            temp = []
            for column in selected_cols:
                # print(X[j,column + 1])
                temp.append(X[j, column + 1])
            # print(temp)
            X_x.append(temp)
        X = np.array(X_x)

    elif(mode == 3):
        selected_cols = IG.IG(X, y, num_k=4)

        X_x = []
        for j in range(0, len(X)):
            temp = []
            for column in selected_cols:
                # print(X[j,column + 1])
                temp.append(X[j, column + 1])
            # print(temp)
            X_x.append(temp)
        X = np.array(X_x)

    return X,y

def labelBinarize(X,y):

    # 二进制化输出，方便后面绘制ROC曲线（用于多标签的OOM数据集）
    temp = []
    for i in range(0, len(y)):
        temp.append(y[i, 0])
    y_set = set(temp)
    y_list = list(temp)
    classes = [i for i in range(0, len(y_set))]
    # print("classes = {},len(classes) = {}".format(classes,len(classes)))
    # classes = list(y_set)
    print("y = ", y_list)
    y = label_binarize(y_list, classes=classes)
    print("y_n = ", y)
    n_classes = y.shape[1]

    return X,y,n_classes