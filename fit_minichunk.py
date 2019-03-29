#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import numpy as np
import os
import pandas as pd
import scipy.signal as sg
import tqdm
from keras.models import load_model, Model, Sequential

import dask.dataframe as dd

n_model = 'model-ep011-loss1.492-val_loss2.455'
chunk_folder = 'chunk_data/'
spect_folder = 'spect_data/'

modelfile = 'models/' + n_model + '.h5'
convnet = load_model(modelfile)
convnet2 = Sequential()
for layer in convnet.layers[:-4]:   # CHANGE THIS INT
    convnet2.add(layer)
convnet2.summary()
segments = 4194
print('nsegs = ',segments)
y_vals = np.load('train_y.npy')

for segment in tqdm.tqdm(range(segments)):

    for i in range(10):

        minichunk_y = y_vals[segment*10 + i]

        chunk_directory = chunk_folder + str(segment) + '/'
        spect_directory = spect_folder + str(segment) + '/'

        filename_n_chunk = chunk_directory + str(i) + 'n.npy'
        filename_p_chunk = chunk_directory + str(i) + 'p.npy'
        if not os.path.exists(chunk_directory):
            os.makedirs(chunk_directory)

        filename_n_spect = spect_directory + str(i) + 'n.npy'

        minichunk = np.array([np.load(filename_n_spect)])
        feature = convnet2.predict(minichunk, verbose=0)

        np.save(filename_n_chunk,feature)
        np.save(filename_p_chunk,minichunk_y)