from keras.layers import Dense
import keras
from keras.models import Sequential

def ann(X_train, X_test, y_train, y_test, mode=False):

    '''
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param mode: mode == True对应OOM数据集，mode == False对应SCM数据集
    :return:
    '''

    classifier = Sequential()
    classifier.add(Dense(output_dim=12, init='uniform', activation='relu', input_dim=len(X_train[0])))
    # Adding the second hidden layer
    classifier.add(Dense(output_dim=12, init='uniform', activation='tanh'))

    classifier.add(Dense(output_dim=12, init='uniform', activation='tanh'))

    classifier.add(Dense(output_dim=12, init='uniform', activation='tanh'))

    classifier.add(Dense(output_dim=12, init='uniform', activation='tanh'))
    # Adding the output layer
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    # Compiling the ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100, verbose=0)
    # Part 3 - Making the predictions and evaluating the model

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int).ravel()

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    # y_test = y_test.astype('int')
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0, 0] + cm[1, 1]) / len(X_test) * 100
    return accuracy, y_pred