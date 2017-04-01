import numpy as np

def GetScaleAbs(train_and_valid_data_to_scale):
    minusMax=np.zeros((20))
    plusMax=np.zeros((20))
    scaling_vector=np.zeros((20))
    for i in range(0, train_and_valid_data_to_scale['input'].shape[0]):
        for j in range(0, train_and_valid_data_to_scale['input'].shape[1]):
            for k in range(0, train_and_valid_data_to_scale['input'].shape[2]):             
                if train_and_valid_data_to_scale['input'][i][j][k] < minusMax[j%20]:
                    minusMax[j%20] = train_and_valid_data_to_scale['input'][i][j][k]
                if train_and_valid_data_to_scale['input'][i][j][k] > plusMax[j%20]:
                    plusMax[j%20] = train_and_valid_data_to_scale['input'][i][j][k] 
    for i in range(0, minusMax.shape[0]):
        if (minusMax[i]*(-1)) > plusMax[i]:
            scaling_vector[i] = minusMax[i]*(-1)
        else:
            scaling_vector[i] = plusMax[i]
    return scaling_vector
def ScaleData(DataToScale, ScalingVector):
    ScaledData = np.zeros(DataToScale.shape)
    for i in range(0, DataToScale.shape[0]):
        for j in range(0, DataToScale.shape[1]):
            for k in range(0, DataToScale.shape[2]):
                ScaledData[i][j][k] = DataToScale[i][j][k] / ScalingVector[j]                 
    return ScaledData

def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result

def create_class_weight(labels_dict):
    total = sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
    	score = total/float(labels_dict[key])
    	class_weight[key] = score

    return class_weight

def delete_column(y_enc):
    y_enci =[]
    for i in (range(len(y_enc))):
    	a=y_enc[i]
    	v=np.array(a)
    	v=np.delete(v, 0, 1)
    	y_enci.append(v)
    return y_enci
