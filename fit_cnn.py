import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Flatten
from keras.models import Model, Sequential, load_model

from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from spectro_gen import i_gen

a = i_gen(batch=40)
b = a.train_gen()
c = a.test_gen()
train_gen = b()
test_gen = c()

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

#(300, 416)
pad = 'same' #change this to same or valid

input_img = Input(shape=(300, 416, 1))
model = Sequential()
model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(300, 416, 1), padding=pad))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, kernel_size=3, activation='relu', padding=pad))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64, kernel_size=3, activation='relu', padding=pad))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(128, kernel_size=3, activation='relu', padding=pad))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(128, kernel_size=3, activation='relu', padding=pad))

model.add(Flatten())
model.add(Dense(1000))
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

#adam params
lr = 0.0001          ################ CHECK ME! default 0.001 ################
beta_1 = 0.9
beta_2 = 0.999
epsilon = 0.0001
decay = 0.0
amsgrad = False

ad = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)

# n_model = 'model-ep003-loss2.441-val_loss2.289'
# modelfile = 'models/' + n_model + '.h5'
# convnet = load_model(modelfile)

model.compile(optimizer=ad, loss='mean_absolute_error', metrics=['mae'])
plot_model(model, to_file='model.png', show_shapes=True)

checkfile = 'models/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
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



