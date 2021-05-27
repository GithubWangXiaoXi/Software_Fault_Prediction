from sklearn.metrics import roc_curve, auc
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from itertools import cycle
import math

def ROCPlot(n_classes,y_test,y_score,title):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # for i in range(n_classes):
    n_classes = len(y_score[0])
    for i in range(n_classes):
        print("y_test[:, i] = ", y_test[:, i])
        print("y_score[:, i] = ", y_score[:, i])
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算微平均ROC曲线和AUC
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 计算宏平均ROC曲线和AUC

    # 首先汇总所有FPR
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # 然后再用这些点对ROC曲线进行插值
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # 最后求平均并计算AUC
    mean_tpr /= n_classes


    # 绘制所有ROC曲线
    plt.figure()
    lw = 2

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    avg_roc = 0
    count = 0
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
        if(math.isnan(roc_auc[i]) == False):
            avg_roc += roc_auc[i]
            count += 1
    avg_roc = round(avg_roc / count,2)

    title = title + " " + str(avg_roc)

    plt.title(title)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

def ROCPlot1(y_test,y_pred,title):

    '''绘制SCM数据集的ROC'''
    tpr, fpr, thresholds = roc_curve(y_test, y_pred)

    import matplotlib.pyplot as plt
    # plt.plot(fpr, tpr)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='blue', lw=2,
             label='ROC curve of class(area = {0:0.2f})'.format(roc_auc))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    title = title + " " + str(round(roc_auc,2))
    plt.title(title)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.show()