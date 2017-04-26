from __future__ import print_function

import keras
import generateImageSets
from keras.models import Sequential, Input
from keras.layers import Dense, Activation, TimeDistributed, SeparableConv2D
from keras.layers import SimpleRNN, Conv2D, LSTM, Embedding, MaxPool2D, Dropout
from keras.layers import MaxPool3D, Reshape
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras import initializers
from keras.optimizers import RMSprop, adam

from PIL import Image
import keras.callbacks
import numpy as np

batch_size = 32
epochs = 3000
hidden_units = 100
attend_size=2

learning_rate = 1e-6
clip_norm = 1.0

height = 32
width = 32
channels = 3

# the data, shuffled and split between train and test sets
dataset = generateImageSets.generate_GT_HR_attention_sets(path="../../dataset/", steps=attend_size)
x_train = dataset[:-10,:attend_size+1,:,:,:]; y_train = dataset[:-10,attend_size+1:,:,:,:]
x_test = dataset[-10:,attend_size+1,:,:,:]; y_test = dataset[-10:,attend_size+1:,:,:,:]
#x_train = x_train.reshape(x_train.shape[0],height,width,channels)

# x_gt has all even column, x_hr has all odd, or vice-versa
# x_gt = x_train[:,0,:,:,:]
# x_hr = x_train[:,1,:,:,:]
# x_train = np.insert(x_hr, np.arange(32), x_gt, axis=2)
#x_train = x_train.reshape(x_train.shape[0],1,height,width,channels)

# x_gt = x_test[:,0,:,:,:]
# x_hr = x_test[:,1,:,:,:]
# x_test = np.insert(x_hr, np.arange(32), x_gt, axis=2)
#x_test = x_test.reshape(x_test.shape[0],1,height,width,channels)

y_train = y_train.reshape(y_train.shape[0],attend_size+1,height,32,channels)
y_test = y_test.reshape(y_test.shape[0],attend_size+1,height,32,channels)
#testVal=x_train[0].reshape(1,height,width,channels)
#input_shape = x_train.shape[1:]

print('Evaluating...')

class PeriodicImageGenerator(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        # Initialization code
        self.epochs = 0

    def on_epoch_end(self, batch, logs={}):
        self.epochs += 1
        if self.epochs % 25 == 0:
            testVal = x_train[np.random.randint(len(x_train)-1)]
            for i in xrange(attend_size+1):
                testVal_ind = testVal[i]
                image = Image.fromarray(testVal_ind.astype('uint8'), 'RGB')
                image.save('image_'+str(self.epochs)+'_'+str(i)+'.jpg')

            testVal=testVal.reshape(1,attend_size+1, height, width, channels)

            val=model.predict(testVal,1,verbose=1)
            val=val.reshape(3,32,32,3)
            for i in xrange(attend_size+1):
                val_ind = val[i]
                image = Image.fromarray(val_ind.astype('uint8'), 'RGB')
                image.save('image_'+str(self.epochs)+'_'+str(i)+'_predicted.jpg')
            # Do stuff like printing metrics

PIG = PeriodicImageGenerator()
model = Sequential()
row, col, pixel = x_train.shape[2:]
row_hidden = 512
col_hidden = 512
# 4D input.
#model.add(Input(shape=(row, col, pixel)))

# Encodes a row of pixels using TimeDistributed Wrapper.
model.add(ConvLSTM2D(filters=3, kernel_size=(1, 1),
                 input_shape=(None, height,width,channels), return_sequences=True))
model.add(TimeDistributed(Conv2D(32, kernel_size=(1, 1),
                 activation='relu'), input_shape=(attend_size+1,height,width,channels)))
model.add(TimeDistributed(Dense(32)))
# Encodes columns of encoded rows.
# model.add(LSTM(col_hidden))
# model.add(Conv2D(32, kernel_size=(1, 1),
#                  activation='relu',
#                  input_shape=(height,width,channels)))
model.add(Dropout(0.15))
#model.add(Reshape(32,))
model.add(MaxPool3D(pool_size=(1,1,1)))
model.add(Conv3D(32, (1, 1, 1), activation='relu'))
model.add(Conv3D(64, (1, 1, 1), activation='relu'))
model.add(Conv3D(128, (1, 1, 1), activation='relu'))
#model.add(SeparableConv3D(128, (1,1,1)))
model.add(Conv3D(64, (1, 1, 1), activation='relu'))
model.add(Conv3D(64, (1, 1, 1), activation='relu'))
model.add(Conv3D(3, (1, 1, 1), padding="same", activation="relu"))

model.add(Dense(3))
model.add(MaxPool3D(pool_size=(1,1,1)))
model.add(Activation('relu'))
#model.add(Embedding(256, output_dim=256))
# model.add(LSTM( 128,
#                 kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
#                 recurrent_initializer=initializers.Identity(gain=1.0),
#                 activation='relu'))
adam = adam(lr=learning_rate)
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='mean_squared_error',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.02,
          callbacks=[PIG])

# Test model
for i in xrange(x_test.shape[0]):
    for j in xrange(attend_size+1):
        testVal = x_test[i,j]
        image = Image.fromarray(testVal.astype('uint8'), 'RGB')
        image.save('image'+str(i)+"_"+str(j)+'_Test.jpg')
    testVal=x_test[i].reshape(1, attend_size+1, height, width, channels)
    val=model.predict(testVal,1,verbose=1)
    val=val.reshape(attend_size+1,32,32,3)
    for j in xrange(attend_size+1):
        testVal = val[j]
        image = Image.fromarray(testVal.astype('uint8'), 'RGB')
        image.save('image'+str(i)+"_"+str(j)+'_Test_predicted.jpg')

# Save out model
model.save('kerasModel_anshul.h5')

# scores = model.evaluate(x_test, y_test, verbose=0)
# print('IRNN test score:', scores[0])
# print('IRNN test accuracy:', scores[1])
