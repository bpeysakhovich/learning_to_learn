import numpy as np
from parameters import par
import pickle
from itertools import product
import matplotlib.pyplot as plt

class Stimulus:

    def __init__(self):

        # we will train the convolutional layers using the training images
        # we will use the train images fro the learning to learn experiments
        self.imagenet_dir = '/home/masse/Context-Dependent-Gating/ImageNet/'
        self.cifar_dir = '/home/bpeysakhovich/Documents/rnn_modeling/learning_to_learn/cifar/cifar-100-python/'
        self.load_cifar_data()

        # for the simple image/saccade task (task 1), select 50 pairs of images
        # TODO: find better name than task1
        self.image_list_task1 = np.random.choice(len(self.test_labels), size = (50,2), replace = False)


    def generate_batch_task1(self, image_pair):

        # 3 outputs: 0 = fixation, 1 = left, 2 = right
        # reward of 0 for maintaining fixation, -1 for improperly breaking fixation
        # reward of 1 for choosing correct action (left/right), reward of -1 otherwise
        # trial stops when agent receives reward not equal to 0

        batch_data   = np.zeros((par['n_time_steps']*par['trials_per_sequence'], par['batch_size'], 32,32,3), dtype = np.float32)
        rewards      = np.zeros((par['n_time_steps']*par['trials_per_sequence'], par['batch_size'], par['n_pol']), dtype = np.float32)
        trial_mask   = np.ones((par['n_time_steps']*par['trials_per_sequence'], par['batch_size']), dtype = np.float32)
        new_trial   = np.zeros((par['n_time_steps']*par['trials_per_sequence']), dtype = np.float32)

        ITI = par['ITI']//par['dt']
        fix = par['fix']//par['dt']
        stim = par['stim']//par['dt']
        delay = par['delay']//par['dt']
        resp = par['resp']//par['dt']

        for i, j in product(range(par['batch_size']), range(par['trials_per_sequence'])):


            start_time = j*par['n_time_steps']
            new_trial[start_time] = 1
            trial_mask[range(start_time,start_time+ITI), :] = 0

            sac_dir = np.random.choice(2)
            image_ind = self.image_list_task1[image_pair, sac_dir]

            batch_data[range(start_time+ITI+fix, start_time+ITI+fix+stim), i, ...] = \
                np.float32(np.reshape(self.test_images[image_ind, :],(1,1,32,32,3), order='F'))/255

            # fixation
            rewards[range(start_time+ITI, start_time+ITI+fix+stim+delay), i, 1] = -1 # fixation break
            rewards[range(start_time+ITI, start_time+ITI+fix+stim+delay), i, 2] = -1 # fixation break
            # response
            rewards[range(start_time+ITI+fix+stim+delay, start_time+par['n_time_steps']), i, 1+sac_dir] = 1 # reward correct response
            rewards[range(start_time+ITI+fix+stim+delay, start_time+par['n_time_steps']), i, 1+(1+sac_dir)%2] = -0.01 # penalize incorrect response

        batch_data += np.random.normal(0, par['noise_in'], size = batch_data.shape)

        return np.maximum(0, batch_data), rewards, trial_mask, new_trial


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


    def load_cifar_data(self):

        """
        Load CIFAR-100 data
        """
        x = pickle.load(open(self.cifar_dir + 'train','rb'), encoding='bytes')

        self.train_images = np.array(x[b'data'])
        self.train_labels = np.array(np.reshape(np.array(x[b'fine_labels']),(-1,1)))

        x = pickle.load(open(self.cifar_dir + 'test','rb'), encoding='bytes')

        self.test_images  = np.array(x[b'data'])
        self.test_labels  = np.array(np.reshape(np.array(x[b'fine_labels']),(-1,1)))

    def generate_image_plus_spatial_batch(self, test = False):

        batch_data   = np.zeros((par['batch_size'], 32,32,3), dtype = np.float32)
        batch_labels = np.zeros((par['batch_size'], par['layer_dims'][-1]), dtype = np.float32)
        #spatial_location   = np.zeros((par['batch_size'], ????????), dtype = np.float32)

        pass


    def generate_image_batch(self, test = False):

        # test refers to drawing images from test data set, or training dataset
        # Select example indices
        random_selection = np.random.randint(0, len(self.train_labels), par['batch_size']) \
            if not test else np.random.randint(0, len(self.test_labels), par['batch_size'])

        num_unique_labels = len(np.unique(self.train_labels))

        # Pick out batch data and labels
        batch_data   = np.zeros((par['batch_size'], 32,32,3), dtype = np.float32)
        batch_labels = np.zeros((par['batch_size'], num_unique_labels), dtype = np.float32)

        for i, image_index in enumerate(random_selection):
            if test:
                k = self.test_labels[image_index] - 1
                batch_labels[i, k] = 1
                batch_data[i, :, :, :] = np.float32(np.reshape(self.test_images[image_index, :],(1,32,32,3), order='F'))/255
            else:
                k = self.train_labels[image_index] - 1
                batch_labels[i, k] = 1
                batch_data[i, :, :, :] = np.float32(np.reshape(self.train_images[image_index, :],(1,32,32,3), order='F'))/255

        return batch_data, batch_labels
