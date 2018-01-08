import os

import numpy as np
import pkg_resources
from keras import backend as K
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.models import Model

from bluewhalecore import BlueWhale
from bluewhalecore import inputlayer
from bluewhalecore import outputlayer
from bluewhalecore.data import DnaBwDataset
from bluewhalecore.data import NumpyBwDataset
from bluewhalecore.data import inputShape
from bluewhalecore.data import outputShape

np.random.seed(1234)

data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')
filename = os.path.join(data_path, 'oct4.fa')
bw_x_train = DnaBwDataset.fromFasta('dna', fastafile=filename, order=1)
bw_y_train = NumpyBwDataset('y', np.random.randint(2, size=(len(bw_x_train),
                                                            1)))


# Option 1:
# Instantiate an ordinary keras model
def kerasmodel():
    input = Input(shape=(4, 200, 1))
    layer = Conv2D(30, (4, 21), activation='relu')(input)
    layer = GlobalAveragePooling2D()(layer)
    output = Dense(1, activation='sigmoid')(layer)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adadelta', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# For comparison, here is how the model would train without BlueWhale
K.clear_session()
np.random.seed(1234)
m = kerasmodel()
h = m.fit(bw_x_train, bw_y_train, epochs=30, batch_size=1000,
          shuffle=False, verbose=0)
print('Option 1')
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('#' * 40)


# Option 2:
# Instantiate an ordinary keras model
def bluewhalemodel():
    input = Input(shape=(4, 200, 1), name='dna')
    layer = Conv2D(30, (4, 21), activation='relu')(input)
    layer = GlobalAveragePooling2D()(layer)
    output = Dense(1, activation='sigmoid', name='y')(layer)
    model = BlueWhale(inputs=input, outputs=output,
                      name='oct4_cnn')
    model.compile(optimizer='adadelta', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


K.clear_session()
m = bluewhalemodel()
h = m.fit(bw_x_train, bw_y_train, epochs=30, batch_size=1000,
          shuffle=False, verbose=0)
print('Option 2')
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('#' * 40)


# Option 3:
# Instantiate an ordinary keras model
@inputlayer
@outputlayer
def bluewhalebody(input, inp, oup, params):
    layer = Conv2D(30, (inp['dna']['shape'][2], 21),
                   activation='relu')(input[0])
    layer = GlobalAveragePooling2D()(layer)
    output = Dense(1, activation='sigmoid')(layer)
    return input, output


K.clear_session()
np.random.seed(1234)
m = BlueWhale.fromShape(inputShape(bw_x_train),
                        outputShape(bw_y_train, 'binary_crossentropy'),
                        'oct4_cnn',
                        modeldef=(bluewhalebody, (10, 'relu',)))
h = m.fit(bw_x_train, bw_y_train, epochs=30, batch_size=1000,
          shuffle=False, verbose=0)
print('Option 3')
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('#' * 40)