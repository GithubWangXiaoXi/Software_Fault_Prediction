import math
from util import loadData
import numpy as np

def IG(class_, features, num_k):

  '''
  :param numpy类型 class_
  :param numpy类型 features
  '''
  '''先将numpy类型转化成list，才能使用set(list)'''
  temp = []
  for i in range(0, len(class_)):
      temp.append(class_[i, 0])

  class_ = temp

  classes = set(class_)

  Hc = 0
  for c in classes:
    pc = class_.count(c)/len(class_)
    Hc += - pc * math.log(pc, 2)
  print('Overall Entropy:', Hc)

  feat_select = []
  for i in range(0,len(features[0])):
      feature = features[:,i]
      feature = feature.tolist()
      feature_values = set(feature)

      Hc_feature = 0
      for feat in feature_values:

        pf = feature.count(feat)/len(feature)
        indices = [i for i in range(len(feature)) if feature[i] == feat]
        clasess_of_feat = [class_[i] for i in indices]
        for c in classes:
            pcf = clasess_of_feat.count(c)/len(clasess_of_feat)
            if pcf != 0:
                temp_H = - pf * pcf * math.log(pcf, 2)
                Hc_feature += temp_H
      ig = Hc - Hc_feature
      feat_select.append(ig)

  topk_index = []
  for i in range(0,num_k):
      max_index = np.argmax(feat_select,axis=0)
      feat_select[max_index] = -100
      topk_index.append(max_index)
  return topk_index

if __name__ == '__main__':
    X,y = loadData.getData("./../../Dataset/OOM/camel-1.6.csv")

    # temp = []
    # for i in range(0,len(y)):
    #     temp.append(y[i,0])
    # # print(temp)

    # y = temp
    # X = X.tolist()
    # print(X)
    # print(y)
    ig = IG(class_=y,features=X,num_k=5)
    print(ig)