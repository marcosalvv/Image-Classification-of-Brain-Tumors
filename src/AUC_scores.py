from sklearn.metrics import roc_auc_score
import numpy as np
from itertools import combinations

dataset = np.load('output_layer/dataset.npy')
#AUC score threshold
auc_th = 0.7
auc_mode = '>'

# Calculate the AUC score of a features with n observations
def auc_func(feature, Y):
    auc = roc_auc_score(Y, feature)
    return auc

X = dataset[:,:-1]
Y = dataset[:,-1]
del dataset

#number of observations
n = X.shape[1]
#classes array
classes_arr = np.unique(Y)
#All the possible combinations of two classes of a multiclass dataset
comb = np.array(list(combinations(classes_arr,2)))
features_best = []
for couple in comb:
    print(f'Calcolo AUC score classe {couple[0]:.0f}vs{couple[1]:.0f}')
    #considerate the dataset with only the observation of two classes
    bin_bool_arr = np.logical_or(Y==couple[0],Y==couple[1])
    X_bin = X[bin_bool_arr]
    Y_bin = Y[bin_bool_arr]
    #Le label are defined as 0 and 1
    Y_bin[Y_bin==couple[0]] = 0
    Y_bin[Y_bin==couple[1]] = 1
    #Calcualte the AUC score for every single features of the data matrix
    auc_arr = np.apply_along_axis(lambda feature: auc_func(feature,Y_bin), 0, X_bin)
    #Select the best features based on the AUC-ROC in the confration between two classes
    if auc_mode=='>':
        feat_sel = np.where(auc_arr>=auc_th)[0]
    elif auc_mode=='<':
        feat_sel = np.where(auc_arr<=auc_th)[0]
    else:
        exit('auc_mode non valido')
    #Add selected features to the best features group
    features_best = np.union1d(features_best, feat_sel)
    #Number of selected features
    features_num = np.size(features_best)
print(f'Le features selezionate:{features_best}')
print(f'Numero di features:{features_num}')

#selected features dataset
sel_data = np.zeros((len(X),features_num))
for idx, features_idx  in enumerate(features_best.astype(int)):
    sel_data[:,idx] = X[:,features_idx]
    sel_dataset = np.column_stack((sel_data,Y))
np.save('output_layer/sel_dataset.npy',sel_dataset)
