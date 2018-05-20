import numpy as np
from parameters import *
import pickle
import matplotlib.pyplot as plt

# TODO: move these parameters to a better home
fix = 20
stim = 20
delay = 10
resp = 10
noise_sd = 0.1
trial_length = fix + stim + delay + resp
par['batch_size'] = 256
par['layer_dims'] = [1000]


class Stimulus:

    def __init__(self):

        # we will train the convolutional layers using the training images
        # we will use the train images fro the learning to learn experiments
        self.imagenet_dir = '/home/masse/Context-Dependent-Gating/ImageNet/'
        self.load_imagenet_data()

        # for the simple image/saccade task (task 1), select 50 pairs of images
        # TODO: find better name than task1
        self.image_list_task1 = np.random.choice(len(self.test_labels), size = (50,2), replace = False)


    def generate_batch_task1(self, image_pair):

        batch_data   = np.zeros((trial_length, par['batch_size'], 32,32,3), dtype = np.float32)
        rewards      = np.zeros((trial_length, par['batch_size'], par['layer_dims'][-1]), dtype = np.float32)

        for i in range(par['batch_size']):
            sac_dir = np.random.choice(2)
            image_ind = self.image_list_task1[image_pair, 0] if sac_dir == 0 else self.image_list_task1[image_pair, 1]
            batch_data[range(fix, fix+stim), i, :, :, :] = np.reshape(self.test_images[image_ind,:,:,:], (1,1,32,32,3))
            # fixation
            rewards[range(0, fix+stim+delay), i,  1] = -1 # fixation break
            rewards[range(0, fix+stim+delay), i, 2] = -1 # fixation break
            # response
            batch_labels[range(fix + stim + delay, trial_length), i, sac_dir] = 1 # reward correct response
            batch_labels[range(fix + stim + delay, trial_length), i, 1+sac_dir%2] = -1 # penalize incorrect response

        batch_data += np.random.normal(0, noise_sd, size = batch_data.shape)

        return np.maximum(0, batch_data), rewards



    def load_imagenet_data(self):

        """
        Load ImageNet data
        """
        self.train_images = np.array([])
        self.train_labels = np.array([])

        for i in range(10):
            x =  pickle.load(open(self.imagenet_dir + 'train_data_batch_' + str(i+1),'rb'))
            self.train_images = np.vstack((self.train_images, x['data'])) if self.train_images.size else x['data']
            labels = np.reshape(np.array(x['labels']),(-1,1))
            self.train_labels = np.vstack((self.train_labels, labels))  if self.train_labels.size else labels

        x =  pickle.load(open(self.imagenet_dir + 'val_data','rb'))
        self.test_images = np.array(x['data'])
        self.test_labels = np.reshape(np.array(x['labels']),(-1,1))


    def generate_cifar_tuning(self):

        """
        Load CIFAR-100 data
        """
        x = pickle.load(open(self.cifar100_dir + 'train','rb'), encoding='bytes')

        self.train_images = np.array(x[b'data'])
        self.train_labels = np.array(np.reshape(np.array(x[b'fine_labels']),(-1,1)))

        x = pickle.load(open(self.cifar100_dir + 'test','rb'), encoding='bytes')

        self.test_images  = np.array(x[b'data'])
        self.test_labels  = np.array(np.reshape(np.array(x[b'fine_labels']),(-1,1)))


    def generate_image_batch(self, test = False):

        # Select example indices
        random_selection = np.random.randint(0, len(self.train_labels), par['batch_size'])

        # Pick out batch data and labels
        batch_data   = np.zeros((par['batch_size'], 32,32,3), dtype = np.float32)
        batch_labels = np.zeros((par['batch_size'], par['layer_dims'][-1]), dtype = np.float32)

        for i, image_index in enumerate(random_selection):
            if test:
                k = self.test_labels[image_index] - 1
                batch_labels[i, k] = 1
                batch_data[i, :] = np.float32(np.reshape(self.test_images[image_index, :],(1,32,32,3), order='F'))/255
            else:
                k = self.train_labels[image_index] - 1
                batch_labels[i, k] = 1
                batch_data[i, :] = np.float32(np.reshape(self.train_images[image_index, :],(1,32,32,3), order='F'))/255

        return batch_data, batch_labels
