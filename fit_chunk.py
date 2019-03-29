import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import numpy as np
import tqdm
from keras.models import load_model, Model, Sequential

n_model = 'model-ep005-loss2.264-val_loss2.037-seed111'
full_chunk_folder = 'full_chunk_data/'

modelfile = 'models/FFstage2/' + n_model + '.h5'
convnet = load_model(modelfile)
convnet2 = Sequential()
for layer in convnet.layers[:-4]:   # CHANGE THIS INT
    convnet2.add(layer)
convnet2.summary()

chunks = 4194
mini_chunks = 10
print("starting loop")

dir_base = "chunk_data/"
out_base = "full_chunk_data/"
for n in tqdm.tqdm(range(chunks)):
    dir_mid = dir_base + str(n) + "/"
    X = np.zeros(1000)
    for m in range(mini_chunks):
        minichunk_x = np.load(dir_mid + str(m) + "n.npy")
        X[m*100:m*100+100] = minichunk_x
    out = out_base + str(n) + ".npy"
    X = np.array([X])
    pred = convnet2.predict(X, verbose=0)
    np.save(out,X)
