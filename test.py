import numpy
import numpy as np
from keras.models import Model
from keras.layers import Dense, concatenate, Input
from keras.layers import LSTM
from keras.utils import np_utils

inp1 = Input(shape=(1,))
inp2 = Input(shape=(1,))
cc1 = concatenate([inp1, inp2], axis=-1) # Merge data must same row column
output = Dense(20, activation='relu')(cc1)
model = Model(inputs=[inp1, inp2], outputs=output)
model.summary()