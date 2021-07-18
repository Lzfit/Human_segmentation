import numpy as np
from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, concatenate, add, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping



def model(shape=(640, 640, 3)):
    """Creates a trainable Unet Model.

    Keyword arguments:
    shape -- shape of the model's input
    """
 
    inputs = Input(shape=shape)

    conv0_0 = Conv2D(64, (3, 3), padding='same')(inputs)
    relu0_1 = Activation('relu')(conv0_0)
    conv0_1 = Conv2D(64, (3, 3), padding='same')(relu0_1)
    relu0_2 = Activation('relu')(conv0_1)
    conv0_2 = Conv2D(32, (2, 2), padding='same')(relu0_2)
    Pooling0 = MaxPooling2D(pool_size=(2, 2))(conv0_2)

    relu1_1 = Activation('relu')(Pooling0)
    conv1_1 = Conv2D(64, (3, 3), padding='same')(relu1_1)
    relu1_2 = Activation('relu')(conv1_1)
    conv1_2 = Conv2D(32, (2, 2), padding='same')(relu1_2)
    plus1 = add([Pooling0, conv1_2]) #plus connection 2
    Pooling21 = MaxPooling2D(pool_size=(2, 2))(plus1)

    relu3_1 = Activation('relu')(Pooling21)
    conv3_1 = Conv2D(64, (3, 3), padding='same')(relu3_1)
    relu3_2 = Activation('relu')(conv3_1)
    conv3_2 = Conv2D(32, (2, 2), padding='same')(relu3_2)
    plus3 = add([Pooling21, conv3_2]) #plus connection 3
    Pooling31 = MaxPooling2D(pool_size=(2, 2))(plus3)

    relu4_1 = Activation('relu')(Pooling31)
    conv4_1 = Conv2D(64, (3, 3), padding='same')(relu4_1)
    relu4_2 = Activation('relu')(conv4_1)
    conv4_2 = Conv2D(32, (2, 2), padding='same')(relu4_2)
    plus4 = add([Pooling31, conv4_2]) #plus connection 4
    Pooling41 = MaxPooling2D(pool_size=(2, 2))(plus4)

    relu5_1 = Activation('relu')(Pooling41)
    conv5_1 = Conv2D(64, (3, 3), padding='same')(relu5_1)
    relu5_2 = Activation('relu')(conv5_1)
    conv5_2 = Conv2D(32, (2, 2), padding='same')(relu5_2)
    plus5 = add([Pooling41, conv5_2]) #plus connection 5
    Pooling51 = MaxPooling2D(pool_size=(2, 2))(plus5)

    relu6_1 = Activation('relu')(Pooling51)
    conv6_1 = Conv2D(64, (3, 3), padding='same')(relu6_1)
    relu6_2 = Activation('relu')(conv6_1)
    conv6_2 = Conv2D(32, (2, 2), padding='same')(relu6_2)
    plus6 = add([Pooling51, conv6_2]) #plus connection 6
    Pooling61 = MaxPooling2D(pool_size=(2, 2))(plus6)

    relu7_1 = Activation('relu')(Pooling61)
    conv7_1 = Conv2D(64, (3, 3), padding='same')(relu7_1)
    relu7_2 = Activation('relu')(conv7_1)
    conv7_2 = Conv2D(32, (2, 2), padding='same')(relu7_2)
    plus7 = add([Pooling61, conv7_2]) #plus connection 7
    Pooling71 = MaxPooling2D(pool_size=(2, 2))(plus7)

    #Expansion Side
    Upsample81 = UpSampling2D(size=(2, 2))(Pooling71)
    relu8_1 = Activation('relu')(Upsample81)
    concat8 = concatenate([relu8_1, plus7], axis=3) #U-connection 1
    conv8_1 = Conv2D(64, (3, 3), padding='same')(concat8)
    relu8_2 = Activation('relu')(conv8_1)
    conv8_2 = Conv2D(32, (2, 2), padding='same')(relu8_2)

    Upsample91 = UpSampling2D(size=(2, 2))(conv8_2)
    relu9_1 = Activation('relu')(Upsample91)
    concat9 = concatenate([relu9_1, plus6], axis=3) #U-connection 2
    conv9_1 = Conv2D(64, (3, 3), padding='same')(concat9)
    relu9_2 = Activation('relu')(conv9_1)
    conv9_2 = Conv2D(32, (2, 2), padding='same')(relu9_2)
    plus8 = add([Upsample91, conv9_2]) #plus connection 8

    Upsample101 = UpSampling2D(size=(2, 2))(plus8)
    relu10_1 = Activation('relu')(Upsample101)
    concat10 = concatenate([relu10_1, plus5], axis=3) #U-connection 3
    conv10_1 = Conv2D(64, (3, 3), padding='same')(concat10)
    relu10_2 = Activation('relu')(conv10_1)
    conv10_2 = Conv2D(32, (2, 2), padding='same')(relu10_2)
    plus9 = add([Upsample101, conv10_2]) #plus connection 9

    Upsample111 = UpSampling2D(size=(2, 2))(plus9)
    relu11_1 = Activation('relu')(Upsample111)
    concat11 = concatenate([relu11_1, plus4], axis=3) #U-connection 4
    conv11_1 = Conv2D(64, (3, 3), padding='same')(concat11)
    relu11_1 = Activation('relu')(conv11_1)
    conv11_2 = Conv2D(32, (2, 2), padding='same')(relu11_1)
    plus10 = add([Upsample111, conv11_2]) #plus connection 10

    Upsample121 = UpSampling2D(size=(2, 2))(plus10)
    relu12_1 = Activation('relu')(Upsample121)
    concat12 = concatenate([relu12_1, plus3], axis=3) #U-connection 5
    conv12_1 = Conv2D(64, (3, 3), padding='same')(concat12)
    relu12_2 = Activation('relu')(conv12_1)
    conv12_2 = Conv2D(32, (2, 2), padding='same')(relu12_2)
    plus11 = add([Upsample121, conv12_2]) #plus connection 11

    Upsample131 = UpSampling2D(size=(2, 2))(plus11)
    relu13_1 = Activation('relu')(Upsample131)
    concat13 = concatenate([relu13_1, plus1], axis=3) #U-connection 6
    conv13_1 = Conv2D(64, (3, 3), padding='same')(concat13)
    relu13_2 = Activation('relu')(conv13_1)
    conv13_2 = Conv2D(32, (2, 2), padding='same')(relu13_2)
    plus12 = add([Upsample131, conv13_2]) #plus connection 12

    Upsample141 = UpSampling2D(size=(2, 2))(plus12)
    relu14_1 = Activation('relu')(Upsample141)
    concat14 = concatenate([relu14_1, conv0_2], axis=3) #U-connection 7
    conv14_1 = Conv2D(64, (3, 3), padding='same')(concat14)
    relu14_2 = Activation('relu')(conv14_1)
    conv14_2 = Conv2D(32, (2, 2), padding='same')(relu14_2)
    plus13 = add([Upsample141, conv14_2]) #plus connection 13
    outputs = Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding='same')(plus13)

    model = Model(inputs, outputs)

    return model


def make_history_plots(history):
    """Creates accuracy and loss history plots for the trained model.

    Keyword arguments:
    history -- History object containing the recorded events in the model training staget
    """
    
    metrics = ['accuracy', 'loss']
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    for i,j in enumerate(metrics):
        ax[i].plot(history.history[j])
        ax[i].plot(history.history['val_' + j])
        ax[i].set_title('model ' + j)
        ax[i].set_ylabel(j)
        ax[i].set_xlabel('epoch')
        ax[i].legend(['train', 'val'], loc='upper left')