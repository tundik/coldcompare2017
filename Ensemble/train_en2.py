# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 14:18:54 2016

@author: CB
"""
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)

import keras.models
from keras.layers import  Dense, merge
from keras.models import Model, load_model
from keras.layers.core import Dropout
import numpy as np
from keras.optimizers import SGD
import pickle
import datetime
import time

import gc
def create_class_weight(labels_dict):
    total = sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = total/float(labels_dict[key])
        class_weight[key] = score

    return class_weight

exec(open("./loadData.py").read())
'''
X_train, y_train = read_data(wavPaths_train, labels_train)
X_devel, y_devel = read_data(wavPaths_devel, labels_devel)
'''


"""
The following section reads and builds pretrained models. We cut the last layer
from the models and merge them to an ensemble model.
We also perform the neccesary data preprocessing steps needed for each model
architecture.
"""
######  model_cnn
model_cnn_path = "models/0.6789087077672682_cnn.h5"
model_cnn = load_model(model_cnn_path)
for layer in model_cnn.layers:
    layer.name = "{}_cnn".format(layer.name)
    layer.trainable = False

# Load and preprocess data for CNN
import h5py
with h5py.File('ds_430.h5', 'r') as hf:
    X_train_cnn = hf['X_train'][:][1:,:].astype(np.float16)
    y_train_cnn = hf['y_train'][:][1:,:].astype(np.float16)
    X_devel_cnn = hf['X_devel'][:][1:,:].astype(np.float16)
    y_devel_cnn = hf['y_devel'][:][1:,:].astype(np.float16)

#Standardize
X_train_mean = X_train_cnn.mean()
X_train_std = X_train_cnn.std()

X_train_cnn -= X_train_mean
X_train_cnn /= X_train_std

X_devel_cnn -= X_train_mean
X_devel_cnn /= X_train_std



######  model_dnn1
modelPath_dnn = "../Class_Weights_PCA_Cold/best_model10.h5"
model_dnn1=load_model(modelPath_dnn)
for layer in model_dnn1.layers:
    layer.trainable = False
    layer.name = "{}_dnn1".format(layer.name)

# Load and preprocess numeric data
from myfunc import encode
from myfunc import create_class_weight
from myfunc import delete_column
from sklearn import preprocessing
from sklearn.decomposition import PCA
print("Data mfcc read started...")
data = pd.read_csv("../ComParE2017_Cold.ComParE.train.arff",delimiter=',',skiprows=range(0, 6379))
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
data2 = pd.read_csv("../ComParE2017_Cold.ComParE.devel.arff",delimiter=',',skiprows=range(0, 6379))
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

y_train_dnn1 = y_train
y_devel_dnn1 = y_val
import pickle
pca=pickle.load(open( "../Class_Weights_PCA_Cold/best_pca10.pickle", "rb" ))
X_train_dnn1 = pca.transform(train_data_input)
X_devel_dnn1 = pca.transform(valid_data_input)
###### End of model_dnn1

label_count={}
for i in range(y_train_dnn1.shape[-1]):
    label_count.update({int(i):len(y_train_dnn1[y_train_dnn1[:,int(i)]==1])})

cweights=create_class_weight(label_count)

outputDim = y_train_cnn.shape[-1]

## Merge CNN and DNN models
# the [-3] drops the last Dense and Dropout layers
merged_layers = merge([model_cnn.layers[-3].output, model_dnn1.layers[-3].output], mode = 'concat')

## Add some feed forward layers
x = Dense(512, activation = 'relu')(merged_layers)
x = Dropout(0.5)(x)
output1 = Dense(outputDim, activation = 'softmax')(x)

model = Model(input=[model_cnn.input, model_dnn1.input], output=output1)

# Early stopping to stop training when the loss is not decreasing anymore
from keras.callbacks import EarlyStopping
from RecallCallback import RecallCallback
recallCallback = RecallCallback()
earlyStopping = EarlyStopping(monitor='val_acc', patience=5, mode='max')

# ModelCheckpointer to save the best weights
from keras.callbacks import ModelCheckpoint
bestWeightsFilePath="best_weights_ensemble.hdf5"
checkpointer = ModelCheckpoint(filepath=bestWeightsFilePath, verbose=1, monitor = "val_acc", save_best_only=True, mode='max')

sgd=SGD(lr=0.001,momentum=0.9, decay=0,nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy']) #best: adadelta

fitting_result=model.fit([X_train_cnn, X_train_dnn1], y_train_cnn, nb_epoch=100, batch_size=30, callbacks=[ earlyStopping, checkpointer], validation_data=([X_devel_cnn, X_devel_dnn1], y_devel_cnn))#,class_weight=cweights)

print("Loading weights with smallest loss")
model.load_weights(bestWeightsFilePath)


predicted_labels = lb.inverse_transform(model.predict([X_devel_cnn, X_devel_dnn1]))
true_labels = lb.inverse_transform(y_devel_cnn)

from sklearn.metrics import classification_report, recall_score
confusionMX=pd.crosstab(true_labels, predicted_labels, rownames=['True'], colnames=['Predicted'], margins=True)
cl_report=classification_report(true_labels, predicted_labels)

print(confusionMX)
print(cl_report)

recall = recall_score(true_labels, predicted_labels, average="macro")
print (recall)

model.save("models/ensemble_{}.h5".format(recall))