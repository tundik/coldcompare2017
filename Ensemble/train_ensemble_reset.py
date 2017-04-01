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

loadDataFromHDF5 = True

trainable = True


import keras.models
from keras.layers import  Dense, merge
from keras.models import Model, load_model
from keras.layers.core import Dropout
import numpy as np
import pickle
import datetime
import time
import h5py
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
#model_cnn_path = "models/0.631577343456112_cnn.h5"
model_cnn_path = "models/0.6789087077672682_cnn.h5"
model_cnn = load_model(model_cnn_path)
for layer in model_cnn.layers:
    layer.name = "{}_cnn".format(layer.name)
    layer.trainable = trainable
    #layer.build(layer.input_shape)

# Load and preprocess data for CNN
import h5py
with h5py.File('ds_430.h5', 'r') as hf:
    X_train_cnn = hf['X_train'][:].astype(np.float32)
    y_train_cnn = hf['y_train'][:].astype(np.float32)
    X_devel_cnn = hf['X_devel'][:].astype(np.float32)
    y_devel_cnn = hf['y_devel'][:].astype(np.float32)

#Standardize
X_train_mean = X_train_cnn.mean()
X_train_std = X_train_cnn.std()

X_train_cnn -= X_train_mean
X_train_cnn /= X_train_std

X_devel_cnn -= X_train_mean
X_devel_cnn /= X_train_std

label_count={}
for i in range(y_train_cnn.shape[-1]):
    label_count.update({int(i):len(y_train_cnn[y_train_cnn[:,int(i)]==1])})

cweights=create_class_weight(label_count)
cweights = {0: 3.0, 1: 6.0}
print("Class weights: ")
print(cweights)

######  model_dnn1
modelPath_dnn = "./Class_Weights_PCA_Cold/best_model150.h5"
model_dnn1=load_model(modelPath_dnn)
for layer in model_dnn1.layers:
    layer.trainable = trainable
    layer.name = "{}_dnn1".format(layer.name)
    #layer.build(layer.input_shape)

# Load and preprocess numeric data
if not False:
    from myfunc import encode
    from myfunc import create_class_weight
    from myfunc import delete_column
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    print("Data mfcc read started...")
    data = pd.read_csv("./ComParE2017_Cold.ComParE.train.arff",delimiter=',',skiprows=range(0, 6379), header=None)
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
    data2 = pd.read_csv("./ComParE2017_Cold.ComParE.devel.arff",delimiter=',',skiprows=range(0, 6379), header=None)
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
    pca=pickle.load(open( "./Class_Weights_PCA_Cold/best_pca150.pickle", "rb" ))
    X_train_dnn1 = pca.transform(train_data_input)
    X_devel_dnn1 = pca.transform(valid_data_input)

    '''
    with h5py.File('ds_pca_150.h5', 'w') as hf:
        hf.create_dataset("X_train",  data=X_train_dnn1)
        hf.create_dataset("y_train",  data=y_train_dnn1)
        hf.create_dataset("X_devel",  data=X_devel_dnn1)
        hf.create_dataset("y_devel",  data=y_devel_dnn1)
    '''

else:
    with h5py.File('ds_pca_150.h5', 'r') as hf:
        X_train_dnn1 = hf['X_train'][:].astype(np.float32)
        y_train_dnn1 = hf['y_train'][:].astype(np.float32)
        X_devel_dnn1 = hf['X_devel'][:].astype(np.float32)
        y_devel_dnn1 = hf['y_devel'][:].astype(np.float32)

###### End of model_dnn1


######  model_dnn2
modelPath_dnn = "./Class_Weights_Mutual_Cold/best_model500.h5"
model_dnn2=load_model(modelPath_dnn)
for layer in model_dnn2.layers:
    layer.trainable = trainable
    layer.name = "{}_dnn2".format(layer.name)
    #layer.build(layer.input_shape)

# Load and preprocess numeric data
if not False:
    from myfunc import encode
    from myfunc import create_class_weight
    from myfunc import delete_column
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    print("Data mfcc read started...")
    data = pd.read_csv("./ComParE2017_Cold.ComParE.train.arff",delimiter=',',skiprows=range(0, 6379), header=None)
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
    data2 = pd.read_csv("./ComParE2017_Cold.ComParE.devel.arff",delimiter=',',skiprows=range(0, 6379), header=None)
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

    y_train_dnn2 = y_train
    y_devel_dnn2 = y_val
    import pickle
    mut=pickle.load(open("./Class_Weights_Mutual_Cold/best_mut500.pickle", "rb"))
    X_train_dnn2 = mut.transform(train_data_input)
    X_devel_dnn2 = mut.transform(valid_data_input)



    '''
    with h5py.File('ds_mut_500.h5', 'w') as hf:
        hf.create_dataset("X_train",  data=X_train_dnn2)
        hf.create_dataset("y_train",  data=y_train_dnn2)
        hf.create_dataset("X_devel",  data=X_devel_dnn2)
        hf.create_dataset("y_devel",  data=y_devel_dnn2)
    '''

else:
    with h5py.File('ds_mut_500.h5', 'r') as hf:
        X_train_dnn2 = hf['X_train'][:].astype(np.float32)
        y_train_dnn2 = hf['y_train'][:].astype(np.float32)
        X_devel_dnn2 = hf['X_devel'][:].astype(np.float32)
        y_devel_dnn2 = hf['y_devel'][:].astype(np.float32)

###### End of model_dnn2


### BOAW START

from myfunc import encode
from myfunc import create_class_weight
from myfunc import delete_column
from sklearn import preprocessing
#modelPath_dnn = "./Class_Weights_BOAW_Cold/boaw_0.629942343265_7682_1869.h5"
modelPath_dnn = "./Class_Weights_BOAW_Cold/boaw_0.6331991894841658_5186_4365.h5"
model_boaw=load_model(modelPath_dnn)
for layer in model_boaw.layers:
    layer.trainable = trainable
    layer.name = "{}_boaw".format(layer.name)
    #layer.build(layer.input_shape)

# Load and preprocess numeric data
print("Data mfcc read started...")
data = pd.read_csv("./Class_Weights_BOAW_Cold/ComParE2017_Cold.ComParE.train.BoAW.arff",delimiter=',',skiprows=range(0,2006),header=None)
data=data.as_matrix()
print ("Data mfcc read finished.")
data=data[:,1:2002]
Y_train=[x[2000] for x in data]
x_train=data[:,0:2000]
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
data2 = pd.read_csv("./Class_Weights_BOAW_Cold/ComParE2017_Cold.ComParE.devel.BoAW.arff",delimiter=',',skiprows=range(0,2006),header=None)
data2=data2.as_matrix()
print ("Data mfcc read finished.")
data2=data2[:,1:2002]
Y_val=[x[2000] for x in data2]
x_val=data2[:,0:2000]
y_enc = [[label2ind[ey] for ey in Y_val]]
y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]
y_enci =delete_column(y_enc)
y_val=y_enci[0]
scaler = preprocessing.StandardScaler().fit(x_train)
X_train_boaw = scaler.transform(x_train)
X_devel_boaw = scaler.transform(x_val)

y_train_boaw = y_train
y_devel_boaw = y_val

### BOAW END

outputDim = y_train_cnn.shape[-1]

## Merge CNN and DNN models
# the [-3] drops the last Dense and Dropout layers
merged_layers = merge([model_cnn.layers[-3].output, model_dnn1.layers[-3].output, model_dnn2.layers[-3].output, model_boaw.layers[-3].output], mode='concat')
#merged_layers = merge([model_cnn.layers[-3].output,  model_boaw.layers[-3].output], mode='concat')
#merged_layers = merge([model_cnn.layers[-3].output, model_dnn1.layers[-3].output, model_dnn2.layers[-3].output], mode='concat')

## Add some feed forward layers
x = Dense(1024, activation = 'relu')(merged_layers)
x = Dropout(0.5)(x)
x = Dense(1024, activation = 'relu')(x) #, init='glorot_uniform'
x = Dropout(0.5)(x)
output1 = Dense(outputDim, activation = 'softmax')(x)

model = Model(input=[model_cnn.input, model_dnn1.input, model_dnn2.input, model_boaw.input], output=output1)
#model = Model(input=[model_cnn.input, model_boaw.input], output=output1)
#model = Model(input=[model_cnn.input, model_dnn1.input, model_dnn2.input], output=output1)

for layer in model.layers:
    print(("{} {} -> {}".format(layer.name, layer.input_shape, layer.output_shape)))

    
monitor = 'val_recall'
mode = 'max'
# Early stopping to stop training when the loss is not decreasing anymore
from keras.callbacks import EarlyStopping
from RecallCallback import RecallCallback
recallCallback = RecallCallback()
earlyStopping = EarlyStopping(monitor=monitor, patience=5, mode=mode)

# ModelCheckpointer to save the best weights
from keras.callbacks import ModelCheckpoint
bestWeightsFilePath = "best_weights_ensemble_all4.hdf5"
checkpointer = ModelCheckpoint(filepath=bestWeightsFilePath, verbose=1, monitor=monitor, save_best_only=True, mode=mode)
from keras.optimizers import Adam, SGD, RMSprop, Adadelta
print(model.layers[1].get_weights()[0][0][0])

weights = model.get_weights()
weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
model.set_weights(weights)
print(model.layers[1].get_weights()[0][0][0])

model.compile(loss='categorical_crossentropy', optimizer='sgd',  metrics=['accuracy']) #best: adadelta
print(model.layers[1].get_weights()[0][0][0])

extraSamplesFromDevel = 0# int(len(X_devel_cnn)/4.0)
X_train_cnn = np.vstack([X_train_cnn, X_devel_cnn[:extraSamplesFromDevel, :]])
X_train_dnn1 = np.vstack([X_train_dnn1, X_devel_dnn1[:extraSamplesFromDevel, :]])
X_train_dnn2 = np.vstack([X_train_dnn2, X_devel_dnn2[:extraSamplesFromDevel, :]])
y_train_cnn = np.vstack([y_train_cnn, y_train_cnn[:extraSamplesFromDevel, :]])
X_devel_cnn[extraSamplesFromDevel:, :], X_devel_dnn1[extraSamplesFromDevel:, :], X_devel_dnn2[extraSamplesFromDevel:, :]



model.fit([X_train_cnn, X_train_dnn1, X_train_dnn2, X_train_boaw], y_train_cnn, nb_epoch=1000, batch_size=64,
          callbacks=[recallCallback, earlyStopping, checkpointer],
          validation_data=([X_devel_cnn[extraSamplesFromDevel:, :],
          X_devel_dnn1[extraSamplesFromDevel:, :],
          X_devel_dnn2[extraSamplesFromDevel:, :],
          X_devel_boaw[extraSamplesFromDevel:, :]],
          y_devel_cnn[extraSamplesFromDevel:, :]), class_weight=cweights)

'''
model.fit([X_train_cnn, X_train_boaw], y_train_cnn, nb_epoch=1000, batch_size=64,
          callbacks=[recallCallback, earlyStopping, checkpointer],
          validation_data=([X_devel_cnn[extraSamplesFromDevel:, :],
          X_devel_boaw[extraSamplesFromDevel:, :]],
          y_devel_cnn[extraSamplesFromDevel:, :]), class_weight=cweights)
'''
print("Loading weights with smallest loss") 
model.load_weights(bestWeightsFilePath)

predicted_labels = lb.inverse_transform(model.predict([X_devel_cnn, X_devel_dnn1, X_devel_dnn2, X_devel_boaw]))
#predicted_labels = lb.inverse_transform(model.predict([X_devel_cnn, X_devel_boaw]))
#predicted_labels = lb.inverse_transform(model.predict([X_devel_cnn, X_devel_dnn1, X_devel_dnn2]))
true_labels = lb.inverse_transform(y_devel_cnn)

from sklearn.metrics import classification_report, recall_score
confusionMX=pd.crosstab(true_labels, predicted_labels, rownames=['True'], colnames=['Predicted'], margins=True)
cl_report=classification_report(true_labels, predicted_labels)

print(confusionMX)
print(cl_report)

recall = recall_score(true_labels, predicted_labels, average="macro")
print (recall)

model.save("models/ensemble_reset_all_{}.h5".format(recall))