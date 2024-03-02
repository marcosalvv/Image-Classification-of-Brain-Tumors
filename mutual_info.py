#Remove dependent variables by using mutual information
import numpy as np
import time
from sklearn.feature_selection import mutual_info_regression
#record start time
#start = time.time()

#Mutual information threshold (features with mi>mi_th are eliminated from the dataset)
mi_th = 0.1625
#Threshold to not eliminate the same feature
mi_th_same_feat = 6
#Load selected features with AUC
dataset_AUC = np.load('datasets/datasets_AUC/ENetV2/sel_dataset_output_layer_171.npy')

#Function to determine MI for each feature
def mi_func(X,X_j):
    #Vector containing the mutual information between the features matrix and the i-th feature
    mi = mutual_info_regression(X,X_j)
    return mi
#Division in data and labels
X = dataset_AUC[:,:-1]
Y = dataset_AUC[:,-1]
#Number of features
d = np.shape(X)[1]
#Determine the MI for each feature
mi = np.apply_along_axis(lambda X_j: mi_func(X,X_j), 0, X)
#Selected features (boolean array)
sel_feat = np.any(mi<=mi_th,axis=1)
#Selected features data matrix
X_sel = X[:,sel_feat]
#Selected features dataset by using MI
dataset_AUC_MI = np.column_stack((X_sel,Y))
#Save dataset
np.save('datasets/datasets_AUC_MI/ENetV2/output_layer/prova.npy', dataset_AUC_MI)

###Print###
#Number of selected features
d_sel = np.shape(X_sel)[1]
print(f"Number of selected features: {d_sel}")

# record end time
#end = time.time()
# print the difference between start 
# and end time in milli. secs
#print(f"The time of execution of above program is : {(end-start)/60}m")
