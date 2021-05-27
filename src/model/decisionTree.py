from sklearn.multiclass import OneVsRestClassifier
from util import ROCPlot
from sklearn.tree import DecisionTreeClassifier

def DecisionTree(n_classes,X_train, X_test, y_train, y_test,mode):
    '''
    :param n_classes:
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param mode: mode == True对应OOM数据集，mode == False对应SCM数据集
    :return:
    '''

    if(mode == True):

        classifier = OneVsRestClassifier(DecisionTreeClassifier(criterion='entropy', random_state=None))
        classifier.fit(X_train, y_train)

        y_score = classifier.predict_proba(X_test)

        print("y_score = ", y_score)

        # Making the Confusion Matrix
        # cm = confusion_matrix(y_test, y_pred)
        # accuracy = (cm[0,0]+cm[1,1])/len(X_test)*100

        ROCPlot.ROCPlot(n_classes=n_classes, y_test=y_test, y_score=y_score, title="DTr")
    else:
        classifier = DecisionTreeClassifier(criterion='entropy', random_state=None)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        # Making the Confusion Matrix
        # cm = confusion_matrix(y_test, y_pred)
        # accuracy = (cm[0,0]+cm[1,1])/len(X_test)*100

        ROCPlot.ROCPlot1(y_test=y_test, y_pred=y_pred, title="DTr")


