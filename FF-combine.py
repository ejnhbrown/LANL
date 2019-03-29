# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Flatten
from keras.models import Model, Sequential

from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt
from chunk_gen import c_gen

a = c_gen(batch=20)#batch=10)#, nsamples=3000)
b = a.train_agg_gen()
c = a.test_agg_gen()
train_gen = b()
test_gen = c()

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

#(1000, 1)

model = Sequential()
model.add(Dense(100, input_dim=1138))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

model.summary()
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
plot_model(model, to_file='chunk_model.png', show_shapes=True)

checkfile = 'models/FFcombine/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-seed111.h5'
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=1),
    ModelCheckpoint(checkfile,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1)
]
history = model.fit_generator(
        train_gen,
        steps_per_epoch=a.train_steps,
        validation_steps=a.test_steps,
        shuffle=True,
        epochs=100,
        validation_data=test_gen,
        callbacks=callbacks
        )

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()