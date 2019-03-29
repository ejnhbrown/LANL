import numpy as np
import random


class i_gen():
    def __init__(self, batch = 1,shuffle=True, testratio = 0.2, n_chunks=4192, directory='spect_data/', seed=111):
        random.seed(seed)
        self.batch = batch
        self.shuffle = shuffle
        self.testratio = testratio
        self.n_chunks = n_chunks
        self.directory = directory
        self.n_test = int(testratio*n_chunks)
        self.n_train = n_chunks - self.n_test
        chunklist = [i for i in range(n_chunks)]
        if shuffle == True:
            random.shuffle(chunklist)
        print(chunklist[0],"_________________")
        self.chunk_testlist = chunklist[:self.n_test]
        self.chunk_trainlist = chunklist[self.n_test:]
        self.mini_testlist = [item for sublist in [[j + i for i in range(10)] for j in [k * 10 for k in self.chunk_testlist]] for item in sublist]
        self.mini_trainlist = [item for sublist in [[j + i for i in range(10)] for j in [k * 10 for k in self.chunk_trainlist]] for item in sublist]
        random.shuffle(self.mini_testlist)
        random.shuffle(self.mini_trainlist)
        self.n_test *= 10
        self.n_train *= 10
        self.train_steps = np.floor(len(self.mini_trainlist)/batch)
        self.test_steps = np.floor(len(self.mini_testlist)/batch)


    def train_gen(self):
        return self.train

    def test_gen(self):
        return self.test

    def train(self):

        n_train = 0
        looping = True
        while looping:
            x_arr = [0] * self.batch
            y_arr = [0] * self.batch
            print(self.mini_trainlist[0])

            for batch_no in range(self.batch):

                chunk_idx = self.mini_trainlist[n_train%self.n_train]
                n_ch = int(np.floor(chunk_idx / 10))
                i = chunk_idx - (n_ch * 10)
                directory = self.directory + str(n_ch) + '/'
                filename_n = directory + str(i) + 'n.npy'
                filename_p = directory + str(i) + 'p.npy'
                minichunk = np.load(filename_n)
                mini_y = np.load(filename_p)
                x_arr[batch_no] = minichunk
                y_arr[batch_no] = mini_y
                n_train += 1

            yield (np.array(x_arr), np.array(y_arr))

    def test(self):

        n_test = 0
        looping = True
        while looping:
            x_arr = [0] * self.batch
            y_arr = [0] * self.batch

            for batch_no in range(self.batch):

                chunk_idx = self.mini_testlist[n_test%self.n_test]
                n_ch = int(np.floor(chunk_idx / 10))
                i = chunk_idx - (n_ch * 10)
                directory = self.directory + str(n_ch) + '/'
                filename_n = directory + str(i) + 'n.npy'
                filename_p = directory + str(i) + 'p.npy'
                minichunk = np.load(filename_n)
                mini_y = np.load(filename_p)
                x_arr[batch_no] = minichunk
                y_arr[batch_no] = mini_y
                n_test += 1

            yield (np.array(x_arr), np.array(y_arr))

        # for chunk_idx in self.testlist:
        #     n_chunk = int(np.floor(chunk_idx / 10))
        #     i = chunk_idx - (n_chunk * 10)
        #     directory = 'spect_data/' + str(n_chunk) + '/'
        #     filename_n = directory + str(i) + '.npy'
        #     filename_p = directory + str(i) + '.pkl'
        #     minichunk = np.load(filename_n).reshape(1,300, 416, 1)
        #     with open(filename_p, 'rb') as pickle_file:
        #         mini_y = np.array([[pickle.load(pickle_file)]])
        #     yield (minichunk, mini_y)