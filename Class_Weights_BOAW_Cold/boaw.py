# coding: utf-8

from __future__ import print_function
import os
import numpy as np
from numpy import newaxis

np.random.seed(1337)

import keras

from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Convolution1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model,Sequential
from keras.optimizers import SGD
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,recall_score,accuracy_score,confusion_matrix,roc_curve,roc_auc_score
import sys

import pandas as pd
import random


#Read MFCC data
print("Data mfcc read started...")
data = pd.read_csv("ComParE2017_Cold.ComParE.train.BoAW.arff",delimiter=',',skiprows=range(0,2006),header=None)
#data = pd.read_csv("ComParE2017_Cold.ComParE.train.BoAW.upsampled.arff",delimiter=',',skiprows=range(0,2006),header=None)

data=data.as_matrix()
print ("Data mfcc read finished.")


data=data[:,1:2002]


Y_train=[x[2000] for x in data]

x_train=data[:,0:2000]

labels = ['C','NC']
label2ind = {label: (index + 1) for index, label in enumerate(labels)}
ind2label = {(index + 1): label for index, label in enumerate(labels)}

print (label2ind)


# In[63]:

max_label = max(label2ind.values())+1
y_enc = [[label2ind[ey] for ey in Y_train]]


# In[65]:

def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result

y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]


# In[71]:

y_enc[0]
y_enci =[]
for i in (range(len(y_enc))):
    a=y_enc[i]
    v=np.array(a)
    v=np.delete(v, 0, 1)
    y_enci.append(v)

y_train=y_enci[0]

len(x_train)

#Read MFCC data
print("Data mfcc read started...")
data2 = pd.read_csv("ComParE2017_Cold.ComParE.devel.BoAW.arff",delimiter=',',skiprows=range(0,2006),header=None)
data2=data2.as_matrix()
print ("Data mfcc read finished.")


data2=data2[:,1:2002]


# In[44]:

Y_val=[x[2000] for x in data2]

x_val=data2[:,0:2000]

max_label = max(label2ind.values())+1
y_enc = [[label2ind[ey] for ey in Y_val]]
y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]

y_enci =[]
for i in (range(len(y_enc))):
    a=y_enc[i]
    v=np.array(a)
    v=np.delete(v, 0, 1)
    y_enci.append(v)

y_val=y_enci[0]

len(x_val)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_train)


train_data_input = scaler.transform(x_train)
valid_data_input = scaler.transform(x_val)
test_data_input = scaler.transform(x_val)

#train_data_input = x_train
#valid_data_input = x_val
#test_data_input = x_val


train_data_target = y_train
valid_data_target = y_val
test_data_target = y_val
# In[80]:
print(train_data_input.shape)
print(test_data_input.shape)
print(valid_data_input.shape)

print(train_data_target.shape)
print(test_data_target.shape)
print(valid_data_target.shape)

def create_class_weight(labels_dict):
    total = sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = total/float(labels_dict[key])
        class_weight[key] = score

    return class_weight

label_count={}
for i in range(train_data_target.shape[-1]):
    label_count.update({int(i):len(train_data_target[train_data_target[:,int(i)]==1])})

cweights=create_class_weight(label_count)

earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')


# In[87]:

model = Sequential()
from keras.layers import Convolution1D, MaxPooling1D, Embedding, Dropout

model.add(Dense(1000, input_shape=(2000,), init='glorot_normal'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu', init='glorot_normal'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu', init='glorot_normal'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


# In[88]:

batch_size=200
epochs = 20

#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(train_data_input, train_data_target, nb_epoch=epochs,batch_size=batch_size, callbacks=[earlyStopping], shuffle=True, validation_data = (valid_data_input, valid_data_target),class_weight=cweights)


model.save("boaw.h5")
score = model.evaluate(test_data_input, test_data_target, batch_size=batch_size)
accuracy = score[1]
loss = score[0]
print("Accuracy: ", accuracy, "  Loss: ", loss)

pr = model.predict_classes(test_data_input)
yh = test_data_target.argmax(1)
print("\n")
print (recall_score(pr, yh, average="macro"))