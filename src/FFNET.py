#Image classification by FeedForwardNetwork
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Number of classes
n_classes = 4
#Validation set percentage
val_perc = 0.2
#Dataset path
dataset  = np.load("datasets/datasets_AUC_MI/VGG16/final_dataset_output_layer_27.npy")
#FFNET model path
model = keras.saving.load_model("models/tuned/VGG16/best_model_output_layer.keras")
#Number of epochs
n_epoch = 50
#Reproducibility seed
seed = 55

#Dataset separation in features (x) e target (Y)
X = dataset[:,:-1]
Y = dataset[:,-1]
Y = Y.astype(int)
#Save all initial network weights
Wsave = model.get_weights()
tf.random.set_seed(seed)
#K-fold crossvalidation
k = np.round(1/val_perc).astype(int)
kf = KFold(n_splits=k,shuffle=True,random_state=seed)
acc_train_cross = np.zeros((k,n_epoch))
acc_val_cross = np.zeros((k,n_epoch))
acc_val_best_cross = np.zeros(k)
conf_matr_cross=np.zeros((n_classes,n_classes,k))
for i, train_val_idx in enumerate(kf.split(X)):
    print(f'Crossvalidation: {i+1}/{k}')
    train_idx, val_idx = train_val_idx
    #Training set
    X_train = X[train_idx,:]
    Y_train= Y[train_idx]
    #Validation set
    X_val = X[val_idx,:]
    Y_val = Y[val_idx]
    #Training and validation set scaling
    norm_layer = keras.layers.Normalization()
    norm_layer.adapt(X_train)
    X_train = norm_layer(X_train)
    X_val = norm_layer(X_val)
    #Training and validazione
    checkpoint_cb = keras.callbacks.ModelCheckpoint(f"models/trained/best_trained_cross_{i}",monitor='val_accuracy',save_best_only=True,overwrite=True)
    history = model.fit(X_train, Y_train, epochs=n_epoch,validation_data=(X_val, Y_val),callbacks=[checkpoint_cb],verbose=0)

    #Accuracy for each epoch and k-fold crossvalidation
    acc_train_cross[i,:] = np.array(history.history["accuracy"])
    acc_val_cross[i,:] = np.array(history.history["val_accuracy"])
    #Load the best model fot the k-th crossvalidation
    best_train_model = keras.models.load_model(f"models/trained/best_trained_cross_{i}")
    #Predicted labels of the best model
    Y_val_pred = best_train_model.predict(X_val)
    #Assignment class with the greatest probability
    Y_val_pred = Y_val_pred.argmax(axis=1)
    #Calculate model accuracy
    _, acc_val_best = best_train_model.evaluate(X_val, Y_val)
    #Best accuracy for each crossvalidation
    acc_val_best_cross[i] = acc_val_best
    #Confusion matriz for the k-fold crossvalidation
    conf_matr_cross[:,:,i] = confusion_matrix(Y_val, Y_val_pred)
    #Reset the network weights to the ones used before training
    model.set_weights(Wsave)    


#Mean e std deviation of the accuracies as the epochs change
acc_train_cross_mean = acc_train_cross.mean(axis=0)
acc_val_cross_mean = acc_val_cross.mean(axis=0)
acc_train_cross_std = acc_train_cross.std(axis=0)
acc_val_cross_std = acc_val_cross.std(axis=0)
conf_matr_cross = conf_matr_cross.astype(int)
#Best mean validation accuracy
acc_cross_mean_val_best = acc_val_cross_mean.max()
#Best mean accuracy epoch
epoch_best = acc_val_cross_mean.argmax()
#Best mean accuracy std deviation
acc_val_cross_std_best = acc_val_cross_std[epoch_best]


###Print and Plot###
#Epochs must start from 1 in the graph
epoch = np.arange(n_epoch)+1
epoch_best = epoch_best+1
#Print nymber of layer and number of neuron units
model.summary()
#Print learning rate model
print(f"Learning rate: {keras.backend.eval(model.optimizer.lr)}")
print(f"Best mean accuracy mean: {acc_cross_mean_val_best}\tstd: {acc_val_cross_std_best}\tEpoca: {epoch_best}")
#Plot Accuracy vs Epoch
#Training epochs array
fig, acc_ep_ax = plt.subplots()
acc_ep_ax.plot(epoch, acc_train_cross_mean,label="Training", color='red')
acc_ep_ax.plot(epoch, acc_val_cross_mean,label="Validation", color='blue')
acc_ep_ax.fill_between(epoch, acc_val_cross_mean-(acc_val_cross_std/2),acc_val_cross_mean+(acc_val_cross_std/2), label="Error",alpha=0.1,color="blue")
plt.axvline(x=epoch_best,linestyle='--', linewidth=0.5,color='black')
acc_ep_ax.set_xlabel('Epochs',fontsize=13)  
acc_ep_ax.set_ylabel('Accuracy',fontsize=13)  
acc_ep_ax.set_title("Accuracy vs Epochs",fontsize=18)
acc_ep_ax.legend()
acc_ep_ax.grid(True)
#Plot confusion matrix
for i in np.arange(k):
    #Best accuracy for the crossvalidation
    print(f"Best accuracy for the crossvalidation {i+1}/{k}: {acc_val_best_cross[i]}")
    disp_conf = ConfusionMatrixDisplay(confusion_matrix=conf_matr_cross[:,:,i],display_labels=('Glioma','Meningioma','Normal','Pituitary'))
    disp_conf.plot()
plt.show()
