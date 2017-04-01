
# coding: utf-8

from __future__ import print_function
import os
import numpy as np
from numpy import newaxis
np.random.seed(1337)
import keras

from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Convolution1D, MaxPooling1D, Embedding, Dense,Dropout , Activation
from keras.models import Model,Sequential
import sys
from keras.optimizers import SGD
from sklearn.metrics import classification_report,recall_score,accuracy_score,confusion_matrix,roc_curve,roc_auc_score

import pandas as pd
import random

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from keras.optimizers import RMSprop

from keras.datasets import mnist
from keras.utils import np_utils

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from myfunc import encode
from myfunc import create_class_weight
from myfunc import delete_column




def data():
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import mutual_info_classif

    print("Data mfcc read started...")
    data = pd.read_csv("ComParE2017_Cold.ComParE.train.arff",delimiter=',',skiprows=range(0, 6379))
    data=data.as_matrix()
    print ("Data mfcc read finished.")


    data=data[:,1:6375]

    Y_train=[x[6373] for x in data]
    Y_train[300]

    x_train=data[:,0:6373]
    labels = ['C','NC']
    label2ind = {label: (index + 1) for index, label in enumerate(labels)}
    ind2label = {(index + 1): label for index, label in enumerate(labels)}
    print (label2ind)
    max_label = max(label2ind.values())+1
    y_enc = [[label2ind[ey] for ey in Y_train]]

    y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]


    y_enci =delete_column(y_enc)

    y_train=y_enci[0]

    print("Data mfcc read started...")
    data2 = pd.read_csv("ComParE2017_Cold.ComParE.devel.arff",delimiter=',',skiprows=range(0, 6379))
    data2=data2.as_matrix()
    print ("Data mfcc read finished.")


    data2=data2[:,1:6375]


    Y_val=[x[6373] for x in data2]
    x_val=data2[:,0:6373]

    y_enc = [[label2ind[ey] for ey in Y_val]]
    y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]

    y_enci =delete_column(y_enc)

    y_val=y_enci[0]

    scaler = preprocessing.StandardScaler().fit(x_train)


    train_data_input = scaler.transform(x_train)
    valid_data_input = scaler.transform(x_val)
    test_data_input = scaler.transform(x_val)

    train_output = y_train
    validation_output = y_val
    test_output = y_val

    print(train_data_input.shape)
    print(test_data_input.shape)
    print(valid_data_input.shape)

    print(train_output.shape)
    print(test_output.shape)
    print(validation_output.shape)

    kbest = SelectKBest(score_func=mutual_info_classif, k=5).fit(train_data_input, train_output[:,0])

    train_input = kbest.transform(train_data_input)
    validation_input = kbest.transform(valid_data_input)
    test_input = kbest.transform(test_data_input)
    import pickle     
    pickle.dump( kbest, open( "best_mut5.pickle", "wb" ) )
    label_count={}
    for i in range(train_output.shape[-1]):
        label_count.update({int(i):len(train_output[train_output[:,int(i)]==1])})

    cweights=create_class_weight(label_count)

    return train_input, train_output, validation_input, validation_output, test_input, test_output, cweights

def model(train_input, train_output, validation_input, validation_output, test_input, test_output, cweights):
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

    model = Sequential()


    model.add(Dense({{choice([150,300,500,750,1000])}}, input_shape=(5,), init={{choice(['glorot_normal','glorot_uniform'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense({{choice([150,300,500,750,1000])}}, activation='relu', init={{choice(['glorot_normal','glorot_uniform'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense({{choice([150,300,500,750,1000])}}, activation='relu', init={{choice(['glorot_normal','glorot_uniform'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense(2, activation='softmax'))

    epochs = 100

    model.compile(loss='binary_crossentropy',optimizer={{choice(['rmsprop','adam'])}},metrics=['acc'])
    model.fit(train_input, train_output, nb_epoch=epochs,batch_size={{choice([50,100,150,200,250,300])}}, callbacks=[earlyStopping], shuffle=True, validation_data = (validation_input, validation_output), class_weight=cweights)

    score = model.evaluate(test_input, test_output)
    accuracy = score[1]
    loss = score[0]
    print("Accuracy: ", accuracy, "  Loss: ", loss)

    pr = model.predict_classes(test_input)
    yh = test_output.argmax(1)
    print("\n")
    print (recall_score(yh, pr, average="macro"))
    uar=recall_score(yh, pr, average="macro")
    print (uar)

    return {'loss': -uar, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':

    train_input, train_output, validation_input, validation_output, test_input, test_output, cweights = data()

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=70,
                                          trials=Trials())

print("Evalutation of best performing model:")
print(best_model.evaluate(test_input, test_output))

print (best_run)

pr = best_model.predict_classes(test_input)
yh = test_output.argmax(1)
print (recall_score(yh, pr, average="macro"))
print (classification_report(yh,pr))

best_model.save('best_model5.h5')

