# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 19:21:55 2017

@author: CB
"""
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

import h5py
'''
with h5py.File('ds_200.h5', 'w') as hf:
    hf.create_dataset("X_train",  data=X_train)
    hf.create_dataset("y_train",  data=y_train)
    hf.create_dataset("X_devel",  data=X_devel)
    hf.create_dataset("y_devel",  data=y_devel)
sys.exit()
'''
with h5py.File('ds_430.h5', 'r') as hf:
    X_train = hf['X_train'][:].astype(np.float32)
    y_train = hf['y_train'][:].astype(np.float32)
    X_devel = hf['X_devel'][:].astype(np.float32)
    y_devel = hf['y_devel'][:].astype(np.float32)

X_train_mean = X_train.mean()
X_train_std = X_train.std()

X_train -= X_train_mean
X_train /= X_train_std

X_devel -= X_train_mean
X_devel /= X_train_std

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)



label_count={}
for i in range(y_train.shape[-1]):
    label_count.update({int(i):len(y_train[y_train[:,int(i)]==1])})

cweights=create_class_weight(label_count)
print("Class weights: ")
print(cweights)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD, RMSprop
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D,  MaxPooling2D 

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD, RMSprop
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D,  MaxPooling2D 
from keras.layers import BatchNormalization

activation_conv = 'tanh'
model = Sequential()
model.add(Convolution2D(input_shape=(X_train.shape[-3],X_train.shape[-2],1),
                        nb_filter=64,
                        nb_row=6,
                        nb_col=6,
                        border_mode='valid',
                        #init='lecun_uniform',
                        activation=activation_conv,
                        subsample=(3, 3)
                        ))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(nb_filter=128,
                        nb_row=5,
                        nb_col=5,
                        border_mode='valid',
                        #init='lecun_uniform', 
                        activation=activation_conv
                        ))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(nb_filter=32,
                        nb_row=8,
                        nb_col=8,
                        border_mode='same',
                        #init='lecun_uniform', 
                        activation=activation_conv
                        ))
                        
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) 
                   
# dense layers
from keras.layers.core import Reshape
model.add(Reshape((16, 5*32)))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(output_dim=len(y_train[0]), activation='softmax'))

for layer in model.layers:
    print(("{} {} -> {}".format(layer.name, layer.input_shape, layer.output_shape)))

#from RecallCallback import RecallCallback
#recallCallback = RecallCallback()
bestWeightsPath = 'best_weights_cnn.hdf5'
model.compile(loss='binary_crossentropy',metrics=["accuracy"], optimizer='adadelta') #binary_crossentropy rmsprop adadelta
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
checkpointer = ModelCheckpoint(filepath=bestWeightsPath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#fitting_result=model.fit(X_train, y_train, nb_epoch=100, batch_size=64, callbacks=[earlyStopping, checkpointer],  validation_split=.2, class_weight = cweights)
fitting_result=model.fit(X_train, y_train, nb_epoch=100, batch_size=64, callbacks=[ earlyStopping, checkpointer], validation_data=(X_devel,y_devel), class_weight = cweights)
#model.load_weights(bestWeightsPath)

X_train=[]
y_train=[]
gc.collect()

predicted_labels = lb.inverse_transform(model.predict(X_devel))
true_labels = lb.inverse_transform(y_devel)

from sklearn.metrics import classification_report, recall_score
confusionMX=pd.crosstab(true_labels, predicted_labels, rownames=['True'], colnames=['Predicted'], margins=True)
cl_report=classification_report(true_labels, predicted_labels)

print(confusionMX)
print(cl_report)

recall = recall_score(true_labels, predicted_labels, average="macro")
print (recall)

model.save("models/{}_cnn.h5".format(recall))
