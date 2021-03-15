import tensorflow as tf 
from tensorflow_eeg.eeg_net.eeg_rnn import * 
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import *
from keras.utils import np_utils
from keras.utils import to_categorical
from data.data_util import *
def CNN_LSTM(learning_rate=1e-3, dropout=0.5):
    CNN_LSTM = Sequential() 
    CNN_LSTM.add(Conv2D(filters=22,kernel_size=(13,1),padding='same',activation='elu',input_shape=(500,1,22)))
    CNN_LSTM.add(MaxPooling2D(pool_size=(3,1),padding='same'))
    CNN_LSTM.add(BatchNormalization())
    CNN_LSTM.add(Dropout(dropout))

    # Second block of conv.
    CNN_LSTM.add(Conv2D(filters=44, kernel_size=(13,1), padding='same', activation='elu'))
    CNN_LSTM.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    CNN_LSTM.add(BatchNormalization())
    CNN_LSTM.add(Dropout(dropout))

    CNN_LSTM.add(Conv2D(filters=88, kernel_size=(13,1), padding='same', activation='elu'))
    CNN_LSTM.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    CNN_LSTM.add(BatchNormalization())
    CNN_LSTM.add(Dropout(dropout))

    CNN_LSTM.add(Flatten())
    #CNN_LSTM.add(Dense(80))


    #CNN_LSTM.add(Reshape((1,80)))
    CNN_LSTM.add(Reshape((1,1672)))

    CNN_LSTM.add(LSTM(32, dropout=0.5, recurrent_dropout=0.5, input_shape=(1,1672), return_sequences=True))
    #CNN_LSTM.add(LSTM(16, dropout=0.5, recurrent_dropout=0.5, input_shape=(1,1672), return_sequences=False))

    CNN_LSTM.add(LSTM(16, activation='sigmoid'))
    CNN_LSTM.add(Dense(4, activation='softmax'))
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    CNN_LSTM.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])
    return CNN_LSTM