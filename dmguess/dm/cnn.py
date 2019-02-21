import numpy as np
import os
import keras
import pandas as pd
import matplotlib.pyplot as plt
import sys
import cv2

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,Adam
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from skimage.util import pad
from skimage.transform import resize, rescale
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_last')

charset = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

def prep_image(image, img_rows, img_cols):
    scale = min(1.0, (img_rows/image.shape[1]) if (image.shape[1] > image.shape[0]) else (img_cols/image.shape[0]))
    image = rescale(image, scale, preserve_range=True)
    image = pad(image, ((0, img_rows - image.shape[0]), (0, img_cols - image.shape[1])), mode='constant', constant_values=255)
    _, image = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY)
    return image

def load_data(data, img_rows, img_cols):
    X = []
    Y = []
    for r in data:
        image = r['grayscale']
        scale = min(1.0, (img_rows/image.shape[1]) if (image.shape[1] > image.shape[0]) else (img_cols/image.shape[0]))
        image = rescale(image, scale, preserve_range=True)
        image = pad(image, ((0, img_rows - image.shape[0]), (0, img_cols - image.shape[1])), mode='constant', constant_values=255)
        _, image = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY)
        X.append(image)
        Y.append(charset.index(r['digit']))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def get_CNN(img_rows, img_cols,LossF ,Metrics,Optimizer=Adam(1e-5),DropP1=0.,DropP2=0.,reg=0.000,batch_norm=False,Activation='relu'):
    print('Optimizer: {0}, DropPerct: {1}, Loss: {2}, reg: {3} batch norm: {4}'.format(Optimizer, (DropP1,DropP2), LossF, reg, batch_norm))
    L2 = l2(reg)

    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(64, (5, 5), activation=Activation,kernel_regularizer=L2, input_shape=(img_rows, img_cols,1)))
    if batch_norm: model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), activation=Activation,kernel_regularizer=L2))
    if batch_norm: model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation=Activation,kernel_regularizer=L2))
    if batch_norm: model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DropP1))

    #model.add(Conv2D(64, (3, 3), activation=Activation,kernel_regularizer=L2))
    #if batch_norm: model.add(BatchNormalization())
    #model.add(Conv2D(64, (3, 3), activation=Activation,kernel_regularizer=L2))
    #if batch_norm: model.add(BatchNormalization())
    #model.add(Conv2D(32, (3, 3), activation=Activation,kernel_regularizer=L2))
    #if batch_norm: model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #if batch_norm: model.add(BatchNormalization())
    #model.add(Dropout(DropP1))

    model.add(Flatten())
    model.add(Dense(128, activation=Activation,kernel_regularizer=L2))
    #model.add(Dense(256, activation=Activation,kernel_regularizer=L2))
    if batch_norm: model.add(BatchNormalization())
    model.add(Dropout(DropP2))
    model.add(Dense(10+26+26, activation='softmax',kernel_regularizer=L2))

    model.compile(loss='categorical_crossentropy', optimizer=Optimizer)
    return model

def evaluate(X,y,model,S=0):
    print ('\n validation scores: \n')
    scores = model.evaluate(X,y, batch_size=1, verbose=0)
    if S: txtfile = open('model_eval{0}.txt'.format(i),'w')
    if S: txtfile.write('MODEL EVAL - Patients {0} \r\n'.format(patients))
    for score,metric in zip(scores,model.metrics_names):
        print ('{0} score: {1}'.format(metric,score) )
        if S: txtfile.write('{0} score: {1} \r\n'.format(metric,score))
    if S: txtfile.close()
    return scores,model.metrics_names

def loaded_model(fn):
    return load_model(fn)

def prepared(images, img_rows=32, img_cols=32):
    X = np.array([prep_image(image, img_rows, img_cols) for image in images])
    X = X.reshape(X.shape[0], img_rows, img_cols, 1)
    X = X.astype('float32')
    X /= 255
    return X

def trained_model_on_data(data, img_rows=32, img_cols=32, load_model_from=None):
    X,Y = load_data(data, img_rows, img_cols)
    p90 = int(X.shape[0] * 0.90)
    X = X.reshape(X.shape[0], img_rows, img_cols, 1)
    Y = keras.utils.to_categorical(Y, num_classes=10+26+26)
    print(Y.shape)

    X_train = X[:p90,:]
    y_train = Y[:p90]
    X_test  = X[p90:,:]
    y_test  = Y[p90:]

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # define data preparation
    shift = 0.15
    zoom  = [0.5,1]
    datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=5.,
        width_shift_range=.05,
        height_shift_range=.05,
        shear_range=20*(3.1416/180),
        zoom_range=0.05,
        fill_mode="constant",cval=0,data_format="channels_last")
    datagen.fit(X_train)

    #Create Keras Model
    #Set model Parameters
    if load_model_from is None:
        cnn_model = get_CNN(
            img_rows=img_rows,
            img_cols=img_cols,
            LossF=categorical_crossentropy, # Set the metric for the model's optimization,
            Metrics=[categorical_accuracy], # calculate model accuracy to make sure it's going in the right direction
            Optimizer=SGD( # Adam(lr)
                lr=1.0e-1, # Set learning rate
                decay=1e-6,
                momentum=0.9,
                nesterov=True
            ),
            DropP1=0.04,
            DropP2=0.4,
            reg=0.0005,
            batch_norm=True,
            Activation='relu' # Nonlinear activation
        )
    else:
        cnn_model = loaded_model(load_model_from)

    #Set training parameters
    N_epochs = 50 # Set low for demo purposes
    Batch_Size = 10 # adjust to fit gpu limitations

    # Set a few callbacks to reduce the learning rate once the model starts to level off
    CALLBACKS = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', \
        factor=0.5, patience=12, verbose=1, mode='auto', epsilon=0.01, cooldown=1000, min_lr=0),

        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', \
        factor=0.5, patience=12, verbose=1, mode='auto', epsilon=0.005, cooldown=1000, min_lr=0),

        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', \
        factor=0.5, patience=12, verbose=1, mode='auto', epsilon=0.001, cooldown=1000, min_lr=0),
    ]

    Steps = len(X_train)/Batch_Size
    print("XT Shape" ,X_train.shape)
    print("YT Shape" ,y_train.shape)

    hist = cnn_model.fit_generator(
        datagen.flow(
            X_train, y_train, \
            batch_size=Batch_Size,
            shuffle=True
        ),
        steps_per_epoch=Steps,
        epochs=N_epochs,
        verbose=True,
        callbacks=CALLBACKS,
        validation_data=(X_test,y_test)
    )

    # Sample a few of the data points along with the predicted label as a sanity check
    def sample_results(imgs,labels,model,n=10):
        N = imgs.shape[0]
        samples = np.random.randint(0,high=N,size=n)
        y_pred = model.predict(imgs)
        for s in samples:
            img,lbl,pred=imgs[s],labels[s],y_pred[s]
            print('y_true: {0}, y_pred: {1}'.format(np.argmax(lbl),np.argmax(pred) ) )

    sample_results(X_test,y_test,cnn_model,len(X_test))

    # evaluate the accuracy on the validation/tuning set
    pred_proba = cnn_model.predict(X_test)
    y_pred=np.argmax(pred_proba,axis=1)
    y_true = np.argmax(y_test,axis=1)
    cnn_accuracy = accuracy_score(y_true, y_pred, normalize=True)
    print(cnn_accuracy)
    return cnn_model
