import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.fftpack
import pandas as pd
import scipy.signal as sg
import tqdm
import random
import pickle

chunklist = [i for i in range(41940)]
#random.shuffle(chunklist)

# NOTE: EDIT SO RANDOM MINICHUNK IS SELECTED

for chunk_idx in chunklist:
    n_chunk = int(np.floor(chunk_idx/10))
    i = chunk_idx-(n_chunk*10)
    print(n_chunk,i)
    directory = 'spect_data/' + str(n_chunk) + '/'
    filename_n = directory + str(i) + 'n.npy'
    filename_p = directory + str(i) + 'p.npy'
    minichunk = np.squeeze(np.load(filename_n))
    mini_y = np.load(filename_p)
    print(minichunk.shape)
    print(mini_y)
    plt.imshow(minichunk,cmap = 'inferno')
    title = 'Chunk: ' + str(n_chunk) + ' Minichunk: ' + str(i) + ' | Time to Quake: ' + str(mini_y)
    plt.title(title)
    plt.show()

# for subdir, dirs, files in os.walk('soect_data/'):
