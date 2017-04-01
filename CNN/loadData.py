# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:18:11 2016

@author: czbal
"""

from scipy import io
from scipy.io import wavfile
import numpy as np
np.random.seed(14)
import os
import matplotlib.pyplot as plt
from matplotlib import mlab
import pandas as pd
wavFolderPath = './Cold_dist/wav/'
from sklearn.preprocessing import LabelBinarizer
import gc


class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super(MyLabelBinarizer, self).transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))[:,-2:]
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super(MyLabelBinarizer, self).inverse_transform(Y[:, 0], threshold)
        else:
            return super(MyLabelBinarizer, self).inverse_transform(Y, threshold)

            
def loadWav(filePath):
    print("\rReading {}".format(filePath), end='')
    return np.array(io.wavfile.read(filePath)[1])
    
    
def loadWavs(filePathList):
    data=list()
    for filePath in filePathList:
        data.append(loadWav(filePath))
    return data

    
def audioToSpectrogram(data):
    tempSpec=np.log10(mlab.specgram(data, NFFT=512, noverlap=128, Fs=16000)[0])
    #tempSpec = tempSpec[0:160,:]
    return tempSpec


def spectrogramListToT4(slist, labels=None, N=5*86, saveMultipleFiles=False, filenames=None, classIds=None): # (1*44100)/(1024-512)=86 #TODO fájlnév, labelbinarizer
    print("SpectrogramListToT4...")
    rows= len(slist[0])
    X=np.empty((0,rows,N, 1)) #(samples, rows, cols, channels)
    X=[]
    y=[]
    for i in range(len(slist)): #len(slist)
        print('\r    Processing no. {} / {}'.format(i+1, len(slist)), end='')
        ranges = np.hstack((np.arange(0,len(slist[i][0]),N), len(slist[i][0])))
        for j in range(len(ranges)-1):
            tempSpec = np.empty((rows,N,1), dtype=np.float32)
            if (len(slist[i][0])<N): #data is shorter than N
                tempSpec[:,:,0]=np.hstack((slist[i],np.zeros((rows, N-len(slist[i][0])))))
            elif (ranges[j+1]-ranges[j]<N): #last range
                tempSpec[:,:,0]=slist[i][:,-N:]
            else:
                tempSpec[:,:,0]=slist[i][:,ranges[j]:ranges[j+1]]
            X.append(tempSpec)
            if labels is not None:
                y.append(labels[i])
    print("\nSpectrogramListToT4 finished")
    return np.array(X, dtype=np.float32), np.array(y)


tsvData = pd.read_csv("./Cold_dist/lab/ComParE2017_Cold.tsv", sep='\t')
tsvData_train = tsvData[tsvData.file_name.str.contains("train")]
tsvData_devel = tsvData[tsvData.file_name.str.contains("devel")]

lb = MyLabelBinarizer()
lb.fit(tsvData_train["Cold (upper respiratory tract infection)"])

wavPaths_train = [os.path.join(wavFolderPath, file_name) for file_name in tsvData_train.file_name]
wavPaths_devel = [os.path.join(wavFolderPath, file_name) for file_name in tsvData_devel.file_name]
labels_train = lb.transform(tsvData_train["Cold (upper respiratory tract infection)"])
labels_devel = lb.transform(tsvData_devel["Cold (upper respiratory tract infection)"])

def read_data(paths, labels=None):
    spectrograms = [audioToSpectrogram(loadWav(wavPath)) for wavPath in paths]
    
    X,y = spectrogramListToT4(spectrograms, labels, N=430)
    lens=[]
    for sg in spectrograms:
        lens.append(len(sg[0]))
    lens=np.array(lens)
    print("Spectrogram length Min: {}, Max: {}, Mean: {}, Std: {}".format(lens.min(), lens.max(), lens.mean(), lens.std()))
    spectrograms=[]
    gc.collect()
    return X,y

#X_train, y_train = read_data(wavPaths_train, labels_train)
#wav=loadWav(wavPaths_train[0])
