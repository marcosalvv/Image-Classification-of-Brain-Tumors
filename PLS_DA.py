import numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Number of classes
n_classes = 4
#Validation set percentage
val_perc = 0.2
#dataset path
dataset  = np.load("datasets/datasets_AUC_MI/VGG16/final_dataset_layer2_99.npy")
#Reproducibility seed
seed = 800

#Converts outputs in categorical vectors
def bin_array(i):   
    Y_arr = np.zeros(n_classes)
    Y_arr[i] = 1
    return Y_arr

#Dataset separation in data(X) e target(Y)
X = dataset[:,:-1]
Y = dataset[:,-1]
Y = Y.astype(int)
#Converts targets in categorical vectors
Y_cat = np.array(list(map(bin_array,Y)))
#Number of observations
n = X.shape[0]
#Number of features
d = X.shape[1]
#K-fold crossvalidation
k = np.round(1/val_perc).astype(int)
kf = KFold(n_splits=k,shuffle=True,random_state=seed)
#For loop inizialition
lv_arr = np.arange(d)
acc_train_cross = np.zeros((k,d))
acc_val_cross = np.zeros((k,d))
n_comp_cross_best = np.zeros(k)
conf_matr =  np.zeros((n_classes,n_classes,k,d))
conf_matr_cross =  np.zeros((n_classes,n_classes,k))
#Execute PLS regression as it varies the number of latent variables
for i, train_val_idx in enumerate(kf.split(X)):
    print(f"Crossvalidazione {i+1}/{k}")
    train_idx, val_idx = train_val_idx
    #Training set
    X_train = X[train_idx,:]
    Y_train = Y[train_idx]
    Y_cat_train = Y_cat[train_idx,:]
    #Validation set
    X_val = X[val_idx,:]
    Y_val = Y[val_idx]
    Y_cat_val = Y_cat[val_idx,:]
    #Standard scaler of input data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    for lv in lv_arr:
        pls2 = PLSRegression(n_components=lv+1,scale=False,max_iter=500,tol=1e-6)
        #Training
        pls2.fit(X_train, Y_cat_train)
        #Validazione
        Y_cat_train_pred = pls2.predict(X_train)
        Y_cat_val_pred = pls2.predict(X_val)
        #Label assignment (nearest value to 1)
        Y_cat_train_pred = np.abs(Y_cat_train_pred-1)
        Y_train_pred = Y_cat_train_pred.argmin(axis=1)
        Y_cat_val_pred = np.abs(Y_cat_val_pred-1)
        Y_val_pred = Y_cat_val_pred.argmin(axis=1)
        #Calculate accuracy
        acc_train = accuracy_score(Y_train, Y_train_pred)
        acc_val = accuracy_score(Y_val, Y_val_pred)
        #Accuracy array for k-fold crossvalidations
        acc_train_cross[i,lv] = acc_train
        acc_val_cross[i,lv] = acc_val
        #Confusion matrices for the latent variables 
        conf_matr[:,:,i,lv] = confusion_matrix(Y_val, Y_val_pred)

#Mean accuracy for k-fold crossvalidation
acc_train_cross_mean = acc_train_cross.mean(axis=0) 
acc_val_cross_mean = acc_val_cross.mean(axis=0)
#Best validation mean accuracy
acc_val_cross_mean_best = acc_val_cross_mean.max()
n_lv_mean_best = acc_val_cross_mean.argmax()
#Standard deviation of the k-fold crossvalidation accuracies
acc_val_cross_std = acc_val_cross.std(axis=0)
#Best mean accuracy standard deviation
acc_val_cross_std_best = acc_val_cross_std[n_lv_mean_best]

#Best accuracies k-fold crossvalidation
acc_val_cross_best = acc_val_cross.max(axis=1)
n_lv_cross_best = acc_val_cross.argmax(axis=1)
#Confusion matrices for the latent variables
for i,lv in enumerate(n_lv_cross_best):
    conf_matr_cross[:,:,i] = conf_matr[:,:,i,lv]
conf_matr_cross = conf_matr_cross.astype(int)

###Print and Plot###
#In the graph le components must start from 1
lv_arr = lv_arr+1
n_lv_mean_best = n_lv_mean_best+1
#Best mean validation accuracy
print(f"Miglior accuracy media:{acc_val_cross_mean_best}\tstd:{acc_val_cross_std_best}\tVariabile latente n°:{n_lv_mean_best}")
#Plot Accuracy vs Epoch
fig, acc_comp_ax = plt.subplots()
acc_comp_ax.plot(lv_arr, acc_train_cross_mean,label="Training", color='red')
acc_comp_ax.plot(lv_arr, acc_val_cross_mean,label="Validation", color='blue')
acc_comp_ax.fill_between(lv_arr, acc_val_cross_mean-(acc_val_cross_std/2),
                       acc_val_cross_mean+(acc_val_cross_std/2), label="Error",
                       alpha=0.1,color="blue")
plt.axvline(x=n_lv_mean_best,linestyle='--', 
            linewidth=0.5,color='black')
acc_comp_ax.set_xlabel('n° latent variables',fontsize=13)  
acc_comp_ax.set_ylabel('Accuracy',fontsize=13)  
acc_comp_ax.set_title("Accuracy vs n° LV",fontsize=18)
acc_comp_ax.legend()
acc_comp_ax.grid(True)
#Confusion matrix plot
for i in np.arange(k):
    #Best accurayc for each crossvalidation
    print(f"Miglior accuracy della crossvalidazione {i+1}/{k}: {acc_val_cross_best[i]}")
    disp_conf = ConfusionMatrixDisplay(confusion_matrix=conf_matr_cross[:,:,i],
        display_labels=('Glioma','Meningioma','Normal','Pituitary'))    
    disp_conf.plot()
plt.show()
