# Search the best FFNET model with an hyperparameters tuning algorithm called Hyperband
import numpy as np
import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split

#Number of classes
n_classes = 4
#Validation set percentage
val_perc = 0.2
#Load dataset
dataset  = np.load("datasets/datasets_AUC_MI/VGG16/final_dataset_layer2_99.npy")
#Hyperparameters
#Number of hidden layers (min e max)
n_hidd_min = 1
n_hidd_max = 5
#Number of neurons per hidden layer (min, max e step)
n_neur_min = 50
n_neur_max = 500
n_neur_step = 5
#Early stopping patience
early_stop_pat = 10
#Optimizer learning (log step)
opt_min = 1e-4
opt_max = 1e-2
#Hypertuning Algorithm
#Max training epochs
max_epoch_alg = 60
#Algorithm iterations
iter_alg=1

#Dataset separation in features (x) and target (Y)
X = dataset[:,:-1]
Y = dataset[:,-1]
Y = Y.astype(int)
#Number of inputs
n_in = np.shape(X)[1]
#Division of dataset in training and test
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_perc,shuffle=True)
#Training and validation set scaling
norm_layer = keras.layers.Normalization()
norm_layer.adapt(X_train)
X_train = norm_layer(X_train)
X_val = norm_layer(X_val)

def build_model(hp,n_in,n_classes):
    #Hyperparameters to optimize
    n_hidden = hp.Int("n_hidden", min_value=n_hidd_min, max_value=n_hidd_max,step=1)
    n_neur = hp.Int("n_neur", min_value=n_neur_min, max_value=n_neur_max,step=n_neur_step)
    learn_rate = hp.Float("learn_rate", min_value=opt_min,max_value=opt_max, sampling="log")
    #FFNET model
    model = keras.Sequential()
    #Input layer
    model.add(keras.layers.Input(shape=n_in))
    #Hidden layers
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neur, activation="relu"))   
    #Output layers
    model.add(keras.layers.Dense(n_classes, activation="softmax"))
    #Optimizer
    opt = keras.optimizers.Adam(learning_rate=learn_rate)
    #Model compilation
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,metrics=['accuracy'])
    return model
#Early stopping
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=early_stop_pat)
#Hyperband tuning
hyperband_tuner = kt.Hyperband(lambda hp: build_model(hp,n_in,n_classes),
                               objective='val_accuracy',
                               max_epochs=max_epoch_alg,
                               factor=3,
                               hyperband_iterations=iter_alg,
                               overwrite=True,
                               directory='tuner_dir',
                               project_name='tuner_model',
                               max_retries_per_trial=5)
hyperband_tuner.search(X_train, Y_train,validation_data=(X_val, Y_val),callbacks=[early_stopping_cb])
#Best hyperparameters
best_hps=hyperband_tuner.get_best_hyperparameters(num_trials=1)[0]
#Best model
model = hyperband_tuner.hypermodel.build(best_hps)
#Save best model
keras.saving.save_model(model,
    filepath = "models/tuned/VGG16/best_model_layer2.keras", overwrite=True)
#Print best hyperparameters
print(best_hps.values)
