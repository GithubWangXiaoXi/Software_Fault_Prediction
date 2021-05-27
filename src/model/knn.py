from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from util import ROCPlot


def Knn(n_classes,X_train, X_test, y_train, y_test,mode):

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
        classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2))

        # classifier.fit(X_train, y_train)

        # Predicting the Test set results
        # y_pred = classifier.predict(X_test)

        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

        # Making the Confusion Matrix
        print("y_score = ", y_score)

        # Making the Confusion Matrix
        # cm = confusion_matrix(y_test, y_pred)
        # accuracy = (cm[0,0]+cm[1,1])/len(X_test)*100

        ROCPlot.ROCPlot(n_classes=n_classes, y_test=y_test, y_score=y_score, title="KNN")
    else:
        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        # Making the Confusion Matrix

        # Making the Confusion Matrix
        # cm = confusion_matrix(y_test, y_pred)
        # accuracy = (cm[0,0]+cm[1,1])/len(X_test)*100

        ROCPlot.ROCPlot1(y_test=y_test, y_pred=y_pred, title="KNN")

