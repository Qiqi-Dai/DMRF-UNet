import os
import sys
import random
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn import preprocessing
from sklearn.preprocessing import scale
import tensorflow as tf
from tensorflow import keras

N_class = 2
N_train = [7200, 9000]
N_test = [800, 1000]
image_sizeX = 128
image_sizeY = 128
num_channel = 1
train_data = []
train_mask1 = []
train_mask2 = []
test_data = []
test_mask1 = []
test_mask2 = []

# Load data
for i in range(N_class):
    for j in range(N_train[i]):
        x = mpimg.imread('./dataset/data/%d/%d.png'%(i+1, j+1))
        x = x.reshape(image_sizeX,image_sizeY,1)
        train_data.append(x)
        m1 = mpimg.imread('./dataset/mask1/%d/%d.png'%(i+1, j+1))
        m1 = m1.reshape(image_sizeX,image_sizeY,1)
        train_mask1.append(m1)
        m2 = mpimg.imread('./dataset/mask2/%d/mask_%d.png'%(i+1, j+1))
        m2 = m2.reshape(image_sizeX,image_sizeY,1)
        train_mask2.append(m2)

    for j in range(N_train[i], N_train[i]+N_test[i]):
        x = mpimg.imread('./dataset/data/%d/%d.png'%(i+1, j+1))
        x = x.reshape(image_sizeX,image_sizeY,1)
        test_data.append(x)
        m1 = mpimg.imread('./dataset/mask1/%d/%d.png'%(i+1, j+1))
        m1 = m1.reshape(image_sizeX,image_sizeY,1)
        test_mask1.append(m1)
        m2 = mpimg.imread('./dataset/mask2/%d/mask_%d.png'%(i+1, j+1))
        m2 = m2.reshape(image_sizeX,image_sizeY,1)
        test_mask2.append(m2)

train_data = np.array(train_data)
train_mask1 = np.array(train_mask1)
train_mask2 = np.array(train_mask2)
test_data = np.array(test_data)
test_mask1 = np.array(test_mask1)
test_mask2 = np.array(test_mask2)

# Build model
def down_block1(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block1(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    us = keras.layers.Conv2D(filters, (2, 2), padding=padding, strides=strides, activation="relu")(us)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck1(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def multi_scale(x, filters, padding="same", strides=1):
    c1 = keras.layers.Conv2D(filters, (1,1), padding=padding, strides=strides, activation="relu")(x)
    c3 = keras.layers.Conv2D(filters, (3,3), padding=padding, strides=strides, activation="relu")(x)
    c5 = keras.layers.Conv2D(filters, (3,3), padding=padding, strides=strides, activation="relu")(x)
    c5 = keras.layers.Conv2D(filters, (3,3), padding=padding, strides=strides, activation="relu")(c5)
    c7 = keras.layers.Conv2D(filters, (3,3), padding=padding, strides=strides, activation="relu")(x)
    c7 = keras.layers.Conv2D(filters, (3,3), padding=padding, strides=strides, activation="relu")(c7)
    c7 = keras.layers.Conv2D(filters, (3,3), padding=padding, strides=strides, activation="relu")(c7)
    c = keras.layers.Concatenate()([c1, c3, c5, c7])
    return c

def down_block2(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    f = int(filters/4)
    c = multi_scale(x, f)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    c = multi_scale(c, f)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block2(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    f = int(filters/4)
    us = keras.layers.UpSampling2D((2, 2))(x)
    us = keras.layers.Conv2D(filters, (2, 2), padding='same', strides=1, activation="relu")(us)
    c = keras.layers.Concatenate()([us, skip])
    c = multi_scale(c, f)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    c = multi_scale(c, f)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck2(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    f = int(filters/4)
    c = multi_scale(x, f)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    c = multi_scale(c, f)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def DMRF_UNet():

    f0 = 64
    f = [f0, f0*2, f0*4, f0*8, f0*16]
    inputs = keras.layers.Input((image_sizeX, image_sizeY, 1))
    
    #### model_1
    p0 = inputs
    c1, p1 = down_block2(p0, f[0])
    c2, p2 = down_block2(p1, f[1])
    c3, p3 = down_block2(p2, f[2])
    c4, p4 = down_block2(p3, f[3])
    
    bn = bottleneck2(p4, f[4])
    
    u1 = up_block2(bn, c4, f[3])
    u2 = up_block2(u1, c3, f[2])
    u3 = up_block2(u2, c2, f[1])
    u4 = up_block2(u3, c1, f[0])
    
    output_1 = keras.layers.Conv2D(1, (1, 1), padding="same", activation="relu")(u4)
    model_1 = tf.keras.Model(inputs, output_1)

    #### model_2
    pp0 = keras.layers.Concatenate()([output_1, inputs])
    cc1, pp1 = down_block2(pp0, f[0])
    cc2, pp2 = down_block2(pp1, f[1])
    cc3, pp3 = down_block2(pp2, f[2])
    cc4, pp4 = down_block2(pp3, f[3])
    
    bnn = bottleneck2(pp4, f[4])
    
    uu1 = up_block2(bnn, cc4, f[3])
    uu2 = up_block2(uu1, cc3, f[2])
    uu3 = up_block2(uu2, cc2, f[1])
    uu4 = up_block2(uu3, cc1, f[0])

    output_2 = keras.layers.Conv2D(1, (1, 1), padding="same", activation="elu")(uu4)
    model_2 = tf.keras.Model(inputs, [output_1, output_2])

    return model_2

# Training
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

model = DMRF_UNet()
model.summary()

total_epoch = 100
batch_size = 100
alpha = 10
beta = 1
path = '/'
model_path = path + 'exp/model.h5'
Adam = keras.optimizers.Adam(lr=1e-4)
lr_metric = get_lr_metric(Adam)
model.compile(optimizer=Adam, loss=['mse', 'mse'], loss_weights=[alpha, beta], metrics=['mae'])
model_checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True)
lr_checkpoint = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=2, min_lr=0)
history = model.fit(x=train_data, y=train_mask1, batch_size=batch_size, epochs=total_epoch, verbose=2, \
    validation_data=(test_data, test_mask1), callbacks=[model_checkpoint])

# Testing
model.load_weights(model_path)
model.evaluate(x=test_data, y=[test_mask1, test_mask2], batch_size=batch_size)
[test_pred1, test_pred2] = model.predict(test_data)
for i in range(len(test_data)):
    pred1 = test_pred1[i].reshape(image_sizeX, image_sizeY) * 255
    pred1 = Image.fromarray(pred1)
    pred1.convert('L').save(path + 'exp/visual/1pred_%d.png'%(i+1))
    pred2 = test_pred2[i].reshape(image_sizeX, image_sizeY) * 255
    pred2 = Image.fromarray(pred2)
    pred2.convert('L').save(path + 'exp/visual/2pred_%d.png'%(i+1))

