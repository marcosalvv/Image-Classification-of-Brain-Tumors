#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
from keras.applications import VGG16
from keras.applications import EfficientNetV2L
from keras import layers, models, Model
#import tqdm as tqdm
from time import sleep
from keras.preprocessing import image
#from PIL import Image
import matplotlib.image as Image

img_size = (256,256)
img_shape = img_size + (3,)
STORE_DIR = 'C:\\Users\\salva\\Desktop\\Output'
DATASET_DIR = 'C:\\Users\\salva\\Desktop\\Tesina_Pattern\\Python\\datasets\\dataset_4classes'

#CNN model (EfficientNetV2L or VGG16)
cnn_type = "EfficientNetV2L"

if cnn_type == "VGG16":
    #features extraction layers
    layer1_V = "block2_pool"
    layer2_V = "block4_pool"

    inputs = tf.keras.Input(shape=(256, 256, 3))
  
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=img_shape)
    base_model.trainable = False

    flatten_layer = layers.Flatten()
    #pooling_layer = layers.GlobalMaxPooling2D()

    #features are extracted at the CNN output 
    model = tf.keras.Sequential()
    model.add(inputs)
    model.add(base_model)
    model.add(flatten_layer)
    #model.add(pooling_layer)
    #number of extracted features
    dim_output_V = model.output_shape[1]

    #features are extracted at two different layers of the model
    model_output1 = base_model.get_layer(layer1_V).output
    m1 = Model(inputs=base_model.input, outputs=model_output1)
    model1 = tf.keras.Sequential()
    model1.add(m1)
    #model1.add(pooling_layer)
    model1.add(flatten_layer)
    #number of extracted features
    dim_layer1_V = model1.output_shape[1]

    model_output2 = base_model.get_layer(layer2_V).output
    m2 = Model(inputs=base_model.input, outputs=model_output2)
    model2 = tf.keras.Sequential()
    model2.add(m2)
    #model2.add(pooling_layer)
    model2.add(flatten_layer)
    #number of extracted features
    dim_layer2_V = model2.output_shape[1]

    #base_model.summary()
    model.summary()

    glioma_path = pathlib.Path(DATASET_DIR + '/glioma_tumor')
    mening_path = pathlib.Path(DATASET_DIR + '/meningioma_tumor')
    normal_path = pathlib.Path(DATASET_DIR + '/normal')
    pituit_path = pathlib.Path(DATASET_DIR + '/pituitary_tumor')

    #images loading
    glioma_imgs = []
    mening_imgs = []
    normal_imgs = []
    pituit_imgs = []

    glioma_file_n = []
    mening_file_n = []
    normal_file_n = []
    pituit_file_n = []

    for file in glioma_path.iterdir():
        img = image.load_img(file)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        glioma_imgs.append(x)
        glioma_file_n.append(file.name)

    for file in mening_path.iterdir():
        img = image.load_img(file)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        mening_imgs.append(x)
        mening_file_n.append(file.name)

    for file in normal_path.iterdir():
        img = image.load_img(file)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        normal_imgs.append(x)
        normal_file_n.append(file.name)

    for file in pituit_path.iterdir():
        img = image.load_img(file)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pituit_imgs.append(x)
        pituit_file_n.append(file.name)

    ########################################
    #remember to choose from which layer to execute the features extraction
    n_features = dim_layer1_V

    features_glioma = np.zeros([glioma_imgs.__len__(), n_features+1])
    features_mening = np.zeros([mening_imgs.__len__(), n_features+1])
    features_normal = np.zeros([normal_imgs.__len__(), n_features+1])
    features_pituit = np.zeros([pituit_imgs.__len__(), n_features+1])

    # Class label assignment 
    features_glioma[:, n_features] = 0
    features_mening[:, n_features] = 1
    features_normal[:, n_features] = 2
    features_pituit[:, n_features] = 3

    header = ""
    for i in range(model.output_shape[1]):
        header += "DeepFeaut%d," % i
    header += "Class"

    print("\n\nFeatures extraction utilizing a CNN"+" "+cnn_type+" "+"...")
    ##################################################################################
    #change the output of model.predict according the model layer (model1 = layer1)

    print("%s extracting..." % "Glioma Tumor")
    sleep(1)
    for idx, img in (enumerate(glioma_imgs)):
        features_glioma[idx][:n_features] = model1.predict(img)
    np.save(STORE_DIR + '/features_glioma_test.npy', features_glioma)
    print("Glioma Tumor completed!")
    sleep(1)

    print("%s extracting..." % "Meningioma Tumor")
    sleep(1)
    for idx, img in (enumerate(mening_imgs)):
        features_mening[idx][:n_features] = model1.predict(img)
    np.save(STORE_DIR + '/features_mening_test.npy', features_mening)
    print("Meningioma Tumor completed!")
    sleep(1)

    print("%s extracting..." % "Normal")
    sleep(1)
    for idx, img in (enumerate(normal_imgs)):
        features_normal[idx][:n_features] = model1.predict(img)
    np.save(STORE_DIR + '/features_normal_test.npy', features_normal)
    print("Normal completed!")
    sleep(1)
    
    print("%s extracting..." % "Pituitary Tumor")
    sleep(1)
    for idx, img in (enumerate(pituit_imgs)):
        features_pituit[idx][:n_features] = model1.predict(img)
    np.save(STORE_DIR + '/features_pituit_test.npy', features_pituit)
    print("Pituitary Tumor completed!")
    sleep(1)
    print("\n\nFeatures extraction completed!")
    
    features_tot = [features_glioma, features_mening, features_normal,features_pituit]

elif cnn_type == "EfficientNetV2L":
    #features extraction layers
    layer1_E = "block3g_add"
    layer2_E = "block5s_add"

    base_model = EfficientNetV2L(include_top=False, weights="imagenet", input_shape=img_shape)
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(256, 256, 3))
    #pooling_layer = layers.GlobalMaxPooling2D()
    #pooling_layer = layers.GlobalAveragePooling2D()
    flatten_layer = layers.Flatten()

    #features are extracted at the CNN output
    model = tf.keras.Sequential()
    model.add(inputs)
    model.add(base_model)
    model.add(pooling_layer)
    #model.add(flatten_layer)
    #number of features extracted
    dim_output_E = model.output_shape[1]

    #features are extracted at two different layers of the model
    model_output1 = base_model.get_layer(layer1_E).output
    m1 = Model(inputs=base_model.input, outputs=model_output1)
    model1 = tf.keras.Sequential()
    model1.add(m1)
    model1.add(pooling_layer)
    #model1.add(flatten_layer)
    #number of features extracted
    dim_layer1_E = model1.output_shape[1]

    model_output2 = base_model.get_layer(layer2_E).output
    m2 = Model(inputs=base_model.input, outputs=model_output2)
    model2 = tf.keras.Sequential()
    model2.add(m2)
    model2.add(pooling_layer)
    #model2.add(flatten_layer)
    #number of features extracted
    dim_layer2_E = model2.output_shape[1]

    #model.summary()

    glioma_path = pathlib.Path(DATASET_DIR + '/glioma_tumor')
    mening_path = pathlib.Path(DATASET_DIR + '/meningioma_tumor')
    normal_path = pathlib.Path(DATASET_DIR + '/normal')
    pituit_path = pathlib.Path(DATASET_DIR + '/pituitary_tumor')
  
    #images loading  
    glioma_imgs = []
    mening_imgs = []
    normal_imgs = []
    pituit_imgs = []

    glioma_file_n = []
    mening_file_n = []
    normal_file_n = []
    pituit_file_n = []

    for file in glioma_path.iterdir():
        img = image.load_img(file)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        glioma_imgs.append(x)
        glioma_file_n.append(file.name)

    for file in mening_path.iterdir():
        img = image.load_img(file)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        mening_imgs.append(x)
        mening_file_n.append(file.name)

    for file in normal_path.iterdir():
        img = image.load_img(file)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        normal_imgs.append(x)
        normal_file_n.append(file.name)

    for file in pituit_path.iterdir():
        img = image.load_img(file)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pituit_imgs.append(x)
        pituit_file_n.append(file.name)
    
    #######################################################################
    #remember to choose from which layer to execute the features extraction
    n_features = dim_layer1_E

    features_glioma = np.zeros([glioma_imgs.__len__(), n_features+1])
    features_mening = np.zeros([mening_imgs.__len__(), n_features+1])
    features_normal = np.zeros([normal_imgs.__len__(), n_features+1])
    features_pituit = np.zeros([pituit_imgs.__len__(), n_features+1])

    features_glioma[:, n_features] = 0
    features_mening[:, n_features] = 1
    features_normal[:, n_features] = 2
    features_pituit[:, n_features] = 3

    header = ""
    for i in range(model.output_shape[1]):
        header += "DeepFeaut%d," % i
    header += "Class"

    print("\n\nEstrazione delle features da un dataset di immagini utilizzando una CNN"+" "+cnn_type+" "+"...")
    
    ###############################################################################
    #change the output of model.predict according the model layer (model1 = layer1)

    print("%s extracting..." % "Glioma Tumor")
    sleep(1)
    for idx, img in (enumerate(glioma_imgs)):
        features_glioma[idx][:n_features] = model1.predict(img)
    np.save(STORE_DIR + '/features_glioma_ENetV2.npy', features_glioma)
    print("Glioma Tumor completed!")
    sleep(1)

    print("%s extracting..." % "Meningioma Tumor")
    sleep(1)
    for idx, img in (enumerate(mening_imgs)):
        features_mening[idx][:n_features] = model1.predict(img)
    np.save(STORE_DIR + '/features_mening_ENetV2.npy', features_mening)
    print("Meningioma Tumor completed!")
    sleep(1)

    print("%s extracting..." % "Normal")
    sleep(1)
    for idx, img in (enumerate(normal_imgs)):
        features_normal[idx][:n_features] = model1.predict(img)
    np.save(STORE_DIR + '/features_normal_ENetV2.npy', features_normal)
    print("Normal completed!")
    sleep(1)

    print("%s extracting..." % "Pituitary Tumor")
    sleep(1)
    for idx, img in (enumerate(pituit_imgs)):
        features_pituit[idx][:n_features] = model1.predict(img)
    np.save(STORE_DIR + '/features_pituit_ENetV2.npy', features_pituit)
    print("Pituitary Tumor completed!")
    sleep(1)
    print("\n\nFeatures extraction completed!")

    features_tot = [features_glioma, features_mening, features_normal,features_pituit]
