# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Flatten
from keras.models import Model, Sequential

from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from matplotlib import pyplot as plt
from chunk_gen import c_gen

a = c_gen(batch=100)#batch=10)#, nsamples=3000)
b = a.train_gen()
c = a.test_gen()
train_gen = b()
test_gen = c()

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

#(1000, 1)

input_img = Input(shape=(300, 416, 1))

model = Sequential()
model.add(Dense(1000, input_dim=1000))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
# model.add(Dense(800))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(500))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

model.summary()
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
# model.compile(optimizer='SGD', loss='mean_absolute_error', metrics=['mae'], lr=0.01, nesterov=True) # alternative
plot_model(model, to_file='chunk_model.png', show_shapes=True)

checkfile = 'models/FFstage2/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        mode='min',
        verbose=1),
    ModelCheckpoint(checkfile,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1),
    CSVLogger('logtest.csv',
              append=False)
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