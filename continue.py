import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

#import tensorflow as tf

from keras.utils import plot_model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from spectro_gen import i_gen
from pickle import dump

#training params
n_model = 'model-ep003-loss2.441-val_loss2.289'
batch = 40
patience = 8
epochs = 100
seed = 111

#adam params
lr = 0.0001          ################ CHECK ME! ################
beta_1 = 0.9
beta_2 = 0.999
epsilon = 0.0001
decay = 0.0
amsgrad = False


a = i_gen(batch=batch,seed=seed)#batch=10)#, nsamples=3000)
b = a.train_gen()
c = a.test_gen()
train_gen = b()
test_gen = c()


modelfile = 'models/' + n_model + '.h5'
convnet = load_model(modelfile)

convnet.summary()

ad = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)

convnet.compile(optimizer=ad, loss='mean_absolute_error', metrics=['mae'])

checkfile = 'models/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-s=' + str(seed) + "cont" + '.h5'
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
        verbose=1),
    ModelCheckpoint(checkfile,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1),
    CSVLogger('logtest.csv',
              append=True)
]
history = convnet.fit_generator(
        train_gen,
        steps_per_epoch=a.train_steps,
        validation_steps=a.test_steps,
        shuffle=True,
        epochs=epochs,
        validation_data=test_gen,
        callbacks=callbacks
        )

dump(history.history,open('history.pkl','wb'))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()