import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import StandardScaler

#Number of classes
n_classes = 4
#Validation set percentage
val_perc = 0.2
#Folder path
dataset  = np.load("datasets/datasets_AUC_MI/VGG16/final_dataset_output_layer_27.npy")
#Reproducibility seed
seed = 400

#Features dataset
X = dataset[:,:-1]
#Target
Y = dataset[:,-1]
Y = Y.astype(int)
#Number of observations
n = X.shape[0]
#Number of features
d = X.shape[1]
#K-fold crossvalidation
k = np.round(1/val_perc).astype(int)
kf = KFold(n_splits=k,shuffle=True)

#For loop initialization
acc_train_cross = np.zeros(k)
acc_val_cross = np.zeros(k)
conf_matr =  np.zeros((n_classes,n_classes,k))
conf_matr_cross =  np.zeros((n_classes,n_classes,k))

#For loop that applies the crossvalidation
for i,train_val_idx in enumerate(kf.split(X)):
    print(f"Crossvalidazione {i+1}/{k}")
    train_idx, val_idx = train_val_idx
    X_train = X[train_idx,:]
    Y_train = Y[train_idx]
    X_val = X[val_idx,:]
    Y_val = Y[val_idx]
    #Standard scaler of input data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    #LDA
    lda = LinearDiscriminantAnalysis(solver="svd")
    #Training
    lda.fit(X_train, Y_train)
    #Validation
    Y_train_pred = lda.predict(X_train)
    Y_val_pred = lda.predict(X_val)
    #Calculate accuracy
    acc_train = accuracy_score(Y_train, Y_train_pred)
    acc_val = accuracy_score(Y_val, Y_val_pred)
    #Accuracy array
    acc_train_cross[i] = acc_train
    acc_val_cross[i] = acc_val
    #K-fold crossvalidation confusion matrices
    conf_matr[:,:,i] = confusion_matrix(Y_val, Y_val_pred)
conf_matr_cross = conf_matr.astype(int)
#K-fold crossvalidation mean accuracy
acc_train_cross_mean = acc_train_cross.mean()
acc_val_cross_mean = acc_val_cross.mean()
#Variance
acc_train_cross_std = acc_train_cross.std()
acc_val_cross_std = acc_val_cross.std()
#K-fold crossvalidation best accuracy
acc_val_cross_best = acc_val_cross.max()
i_val_cross_best = acc_val_cross.argmax()

#Print and plot
for i in np.arange(k):
    #K-fold crossvalidation best accuracy
    print(f"Crossvalidation accuracy {i+1}/{k}: {acc_val_cross[i]}")
    #Confusion matrices for each crossvalidation
    disp_conf = ConfusionMatrixDisplay(confusion_matrix=conf_matr_cross[:,:,i],
        display_labels=('Glioma','Meningioma','Normal','Pituitary'))    
    disp_conf.plot()
print(f"Mean accuracy: {acc_val_cross_mean}\tstd:{acc_val_cross_std} ")
plt.show()
