import numpy as np
import os
import pandas as pd
import scipy.signal as sg
import tqdm


train = pd.read_pickle('train.pkl')
rows = 150_000
segments = 100
segments = int(np.floor(train.shape[0] / rows))
print('nsegs = ',segments)

mean = -13.138772083030858
std = 1.6273252274961973

binlist = list(np.round(np.linspace(-60, 5, num=651),decimals=1))
histlist = [0*650]

#(301, 4160)

for segment in tqdm.tqdm(range(segments)):
    chunk = train.iloc[segment * rows:segment * rows + rows]
    x = chunk['acoustic_data']
    y = chunk['time_to_failure'].values

    f, t, Sxx = sg.spectrogram(x, 4000000, window=('tukey', 2), nfft=600, noverlap=220)
    Sxx = Sxx[1:]
    Sxx = np.log(Sxx)
    Sxx = np.divide(np.subtract(Sxx, mean), std)

    for i in range(10):

        csize = int(len(Sxx[0])/10.0)
        minichunk = np.array([a[i*csize:(i+1)*csize] for a in Sxx]).reshape(300,416,1)
        directory = 'spect_data/' + str(segment) + '/'
        filename_n = directory + str(i) + 'n.npy'
        filename_p = directory + str(i) + 'p.npy'
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(filename_n,minichunk)
        np.save(filename_p,np.array([y[int((i+1)*(len(y)/10.0))-1]]))

        # with open(filename_p, 'wb') as pickle_file:
        #     pickle.dump(y[int((i+1)*(len(y)/10.0))-1],pickle_file)


        # for voltage in minichunk:
        #     chunk_df.loc[df_index, 'Chunk'] = segment
        #     chunk_df.loc[df_index, 'Minichunk'] = i
        #     chunk_df.loc[df_index, 'Acoustic'] = voltage
        #     #print(df_index)
        #     df_index += 1

# print('Pickling...')
# chunk_df.to_pickle('Spectro_data.pkl')                                                                   