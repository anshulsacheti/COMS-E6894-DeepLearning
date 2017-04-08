from __future__ import print_function

import keras
import generateImageSets
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN, Conv2D, LSTM, Embedding, MaxPool2D
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras import initializers
from keras.optimizers import RMSprop

from PIL import Image
import keras.callbacks

batch_size = 1
epochs = 300
hidden_units = 100

learning_rate = 1e-6
clip_norm = 1.0

# the data, shuffled and split between train and test sets
dataset = generateImageSets.generate_GT_HR_sets("../../dataset/")
x_train = dataset[:,:-1,:,:,:]; y_train = dataset[:,-1,:,:,:]
x_train = x_train.reshape(x_train.shape[0],64,32,3)
testVal=x_train[0].reshape(1,64,32,3)
input_shape = x_train.shape[1:]


print('Evaluate IRNN...')

class PeriodicImageGenerator(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        # Initialization code
        self.epochs = 0

    def on_epoch_end(self, batch, logs={}):
        self.epochs += 1
        if self.epochs % 25 == 0:
            val=model.predict(testVal,1,verbose=1)
            val=val.reshape(32,32,3)
            image = Image.fromarray(val.astype('uint8'), 'RGB')
            image.save('image'+str(self.epochs)+'.jpg')
            # Do stuff like printing metrics

PIG = PeriodicImageGenerator()
model = Sequential()
model.add(Conv2D(32, kernel_size=(1, 1),
                 activation='relu',
                 input_shape=(64,32,3)))
model.add(MaxPool2D(pool_size=(2,1)))
model.add(Conv2D(3, (1, 1), activation='relu'))
#model.add(Embedding(256, output_dim=256))
# model.add(LSTM( 128,
#                 kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
#                 recurrent_initializer=initializers.Identity(gain=1.0),
#                 activation='relu'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='binary_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.02,
          callbacks=[PIG])

# scores = model.evaluate(x_test, y_test, verbose=0)
# print('IRNN test score:', scores[0])
# print('IRNN test accuracy:', scores[1])
