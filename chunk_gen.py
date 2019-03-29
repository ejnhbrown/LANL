import numpy as np
import random


class c_gen():
    def __init__(self, batch = 1,shuffle=True, testratio = 0.2, n_chunks=4192, directory='chunk_data/', seed=111):
        random.seed(seed)
        self.batch = batch
        self.shuffle = shuffle
        self.testratio = testratio
        self.n_chunks = n_chunks
        self.directory = directory
        self.agg_dir = 'agg_data/'      #placeholder
        self.n_test = int(testratio*n_chunks)
        self.n_train = n_chunks - self.n_test
        chunklist = [i for i in range(n_chunks)]
        if shuffle == True:
            random.shuffle(chunklist)
        print(chunklist[0], "_________________")
        self.testlist = chunklist[:self.n_test]
        self.trainlist = chunklist[self.n_test:]
        self.train_steps = np.floor(len(self.trainlist)/batch)
        self.test_steps = np.floor(len(self.testlist)/batch)


    def train_gen(self):
        return self.train

    def test_gen(self):
        return self.test

    def train_agg_gen(self):
        return self.train_agg

    def test_agg_gen(self):
        return self.test_agg

    def train(self):

        n_train = 0
        looping = True
        while looping:
            x_arr = [0] * self.batch
            y_arr = [0] * self.batch
            print(self.trainlist[0])

            for batch_no in range(self.batch):
                n_ch = self.trainlist[n_train % self.n_train]
                directory = self.directory + str(n_ch) + '/'
                features = [0] * 10
                chunk_y = [0] * 10

                for m in range(10):
                    filename_n = directory + str(m) + 'n.npy'
                    filename_p = directory + str(m) + 'p.npy'
                    features[m] = np.load(filename_n)
                    chunk_y[m] = np.load(filename_p)
                features = np.array(features)
                chunk_y = np.array(chunk_y)
                x_arr[batch_no] = features.flatten()
                y_arr[batch_no] = chunk_y[-1]
                n_train += 1

            yield (np.array(x_arr), np.array(y_arr))

    def train_agg(self):

        n_train = 0
        looping = True
        while looping:
            x_arr = [0] * self.batch
            y_arr = [0] * self.batch
            print(self.trainlist[0])

            for batch_no in range(self.batch):
                n_ch = self.trainlist[n_train % self.n_train]
                directory = self.directory + str(n_ch) + '/'
                features = [0] * 10
                chunk_y = [0] * 10

                for m in range(10):
                    filename_n = directory + str(m) + 'n.npy'
                    filename_p = directory + str(m) + 'p.npy'
                    features[m] = np.load(filename_n)
                    chunk_y[m] = np.load(filename_p)
                features = np.array(features)
                chunk_y = np.array(chunk_y)
                x_arr[batch_no] = np.concatenate([features.flatten(),np.load(self.agg_dir + str(n_ch) + '.npy')])
                y_arr[batch_no] = chunk_y[-1]
                n_train += 1

            yield (np.array(x_arr), np.array(y_arr))


    def test(self):

        n_test = 0
        looping = True
        while looping:
            x_arr = [0] * self.batch
            y_arr = [0] * self.batch

            for batch_no in range(self.batch):
                n_ch = self.testlist[n_test % self.n_test]
                directory = self.directory + str(n_ch) + '/'
                features = [0] * 10
                chunk_y = [0] * 10

                for m in range(10):
                    filename_n = directory + str(m) + 'n.npy'
                    filename_p = directory + str(m) + 'p.npy'
                    features[m] = np.load(filename_n)
                    chunk_y[m] = np.load(filename_p)

                features = np.array(features)
                chunk_y = np.array(chunk_y)
                x_arr[batch_no] = features.flatten()
                y_arr[batch_no] = chunk_y[-1]
                n_test += 1

            yield (np.array(x_arr), np.array(y_arr))

    def test_agg(self):

        n_test = 0
        looping = True
        while looping:
            x_arr = [0] * self.batch
            y_arr = [0] * self.batch

            for batch_no in range(self.batch):
                n_ch = self.testlist[n_test % self.n_test]
                directory = self.directory + str(n_ch) + '/'
                features = [0] * 10
                chunk_y = [0] * 10

                for m in range(10):
                    filename_n = directory + str(m) + 'n.npy'
                    filename_p = directory + str(m) + 'p.npy'
                    features[m] = np.load(filename_n)
                    chunk_y[m] = np.load(filename_p)
                features = np.array(features)
                chunk_y = np.array(chunk_y)
                x_arr[batch_no] = np.concatenate([features.flatten(), np.load(self.agg_dir + str(n_ch) + '.npy')])
                y_arr[batch_no] = chunk_y[-1]
                n_test += 1

            yield (np.array(x_arr), np.array(y_arr))