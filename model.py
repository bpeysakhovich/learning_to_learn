import tensorflow as tf
import numpy as np
import task
import matplotlib.pyplot as plt
from parameters import par
from convolutional_layers import apply_convolutional_layers
import os, sys
print('TensorFlow version ', tf.__version__)


# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print('Using EI Network:\t', par['EI'])
print('Synaptic configuration:\t', par['synapse_config'], "\n")

"""
Model setup and execution
"""
class Model:

    def __init__(self, input_data, target_data, actual_reward, pred_reward, actual_action, mask):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=0)
        self.target_data = tf.unstack(target_data, axis=0)
        self.pred_reward = tf.unstack(pred_reward, axis=0)
        self.actual_action = tf.unstack(actual_action, axis=0)
        self.actual_reward = tf.unstack(actual_reward, axis=0)

        self.time_mask = tf.unstack(mask, axis=0)

        # Load the initial hidden state activity to be used at the start of each trial
        self.hidden_init = tf.constant(par['h_init'])

        # Build the TensorFlow graph
        self.rnn_cell_loop(self.hidden_init)

        # Train the model
        self.optimize()


    def rnn_cell_loop(self, h):


        self.W_ei = tf.constant(par['EI_matrix'])
        self.h = [] # RNN activity
        self.pol_out = [] # policy output
        self.val_out = [] # value output
        self.syn_x = [] # STP available neurotransmitter
        self.syn_u = [] # STP calcium concentration

        # we will add the first element to these lists since we need to input the previous action and reward
        # into the RNN
        self.action = []
        self.action.append(tf.constant(np.zeros((par['n_pol'], par['batch_size']), dtype = np.float32)))
        self.reward = []
        self.reward.append(tf.constant(np.zeros((par['n_val'], par['batch_size']), dtype = np.float32)))

        self.mask = []
        self.mask.append(tf.constant(np.ones((1, par['batch_size']), dtype = np.float32)))

        """
        Initialize weights and biases
        """
        with tf.variable_scope('recurrent'):
            W_in0 = tf.get_variable('W_in0', initializer = par['W_in0_init'])
            W_in1 = tf.get_variable('W_in1', initializer = par['W_in1_init'])
            b_in0 = tf.get_variable('b_in0', initializer = par['b_in0_init'])
            W_rnn = tf.get_variable('W_rnn', initializer = par['W_rnn_init'])
            b_rnn = tf.get_variable('b_rnn', initializer = par['b_rnn_init'])
            W_reward_pos = tf.get_variable('W_reward_pos', initializer = par['W_reward_pos_init'])
            W_reward_neg = tf.get_variable('W_reward_neg', initializer = par['W_reward_neg_init'])
            W_action = tf.get_variable('W_action', initializer = par['W_action_init'])
            W_pol_out = tf.get_variable('W_pol_out', initializer = par['W_pol_out_init'])
            b_pol_out = tf.get_variable('b_pol_out', initializer = par['b_pol_out_init'])
            W_val_out = tf.get_variable('W_val_out', initializer = par['W_val_out_init'])
            b_val_out = tf.get_variable('b_val_out', initializer = par['b_val_out_init'])

        """
        Loop through the neural inputs to the RNN, indexed in time
        """
        for rnn_input, target, time_mask in zip(self.input_data, self.target_data, self.time_mask):

            x = apply_convolutional_layers(rnn_input, par['conv_weight_fn'])
            x = tf.transpose(x)

            h, action, pol_out, val_out, mask, reward  = self.rnn_cell(x, h, self.action[-1], self.reward[-1], \
                self.mask[-1], target, time_mask)

            self.h.append(h)
            self.action.append(tf.transpose(action))
            self.pol_out.append(pol_out)
            self.val_out.append(val_out)
            self.mask.append(mask)
            self.reward.append(tf.reshape(reward, [par['n_val'], par['batch_size']]))

        self.mask = self.mask[:-1]
        self.reward = self.reward[1:]
        self.action = self.action[1:]


    def rnn_cell(self, x, h, prev_action, prev_reward, mask, target, time_mask):

        # in TF v1.8, I can use reuse = tf.AUTO_REUSE, and get rid of weight initialization above
        with tf.variable_scope('recurrent', reuse = True):
            W_in0 = tf.get_variable('W_in0')
            W_in1 = tf.get_variable('W_in1')
            b_in0 = tf.get_variable('b_in0')
            W_rnn = tf.get_variable('W_rnn')
            W_reward_pos = tf.get_variable('W_reward_pos')
            W_reward_neg = tf.get_variable('W_reward_neg')
            W_action = tf.get_variable('W_action')
            b_rnn = tf.get_variable('b_rnn')
            W_pol_out = tf.get_variable('W_pol_out')
            b_pol_out = tf.get_variable('b_pol_out')
            W_val_out = tf.get_variable('W_val_out')
            b_val_out = tf.get_variable('b_val_out')

        # pass the output of the convolutional layers through the feedforward layer(s)
        x = tf.nn.relu(tf.matmul(W_in0, x) + b_in0)

        rnn_noise = tf.random_normal([par['n_hidden'], par['batch_size']], 0, par['noise_rnn'], dtype=tf.float32)

        h = tf.nn.relu(h*(1-par['alpha']) + par['alpha']*(tf.matmul(W_in1, x) + tf.matmul(W_rnn, h) \
            + tf.matmul(W_reward_pos, tf.nn.relu(prev_reward)) + tf.matmul(W_reward_neg, tf.nn.relu(-prev_reward)) \
            + tf.matmul(W_action, prev_action) + b_rnn + rnn_noise))

        pol_out = tf.matmul(W_pol_out, h) + b_pol_out
        val_out = tf.matmul(W_val_out, h) + b_val_out
        action_index = tf.multinomial(tf.transpose(pol_out), 1)
        action = tf.one_hot(tf.squeeze(action_index), par['n_pol'])

        reward = tf.reduce_sum(action*target, axis = 1)

        continue_trial = tf.cast(tf.equal(prev_reward, 0.), tf.float32)
        mask *= continue_trial

        return h, action, pol_out, val_out, mask, reward


    def optimize(self):

        epsilon = 1e-7
        """
        Calculate the loss functions and optimize the weights
        """
        Z = tf.reduce_sum(tf.stack([tf.reduce_sum(time_mask*mask) for (mask, time_mask) in zip(self.mask, self.time_mask)]))

        pol_out_soft_max = [tf.nn.softmax(pol_out, dim=0) for pol_out in self.pol_out]
        print('pol_out_soft_max', pol_out_soft_max[0])
        print('pred_reward', self.val_out[0])
        print('actual_reward', self.reward[0])
        print('mask', self.mask[0])
        print('time_mask', self.time_mask[0])
        print('Z', Z)

        self.pol_loss = -1.*tf.reduce_sum(tf.stack([(actual_reward - pred_reward)*time_mask*mask*tf.reduce_sum(act*tf.log(epsilon+pol_out), axis = 0) \
            for (pol_out, val_out, act, mask, time_mask, pred_reward, actual_reward) in zip(pol_out_soft_max, self.val_out, \
            self.actual_action, self.mask, self.time_mask, self.val_out, self.reward)]))/Z

        self.entropy_loss = -1.*tf.reduce_sum(tf.stack([time_mask*mask*pol_out*tf.log(epsilon+pol_out) \
            for (pol_out, mask, time_mask) in zip(pol_out_soft_max, self.mask, self.time_mask)]))/Z


        self.val_loss = tf.reduce_mean(tf.stack([tf.squeeze(mask)*tf.square(val_out - actual_reward) \
                for (val_out, mask, actual_reward) in zip(self.val_out, self.mask, self.actual_reward)]))

        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        spike_loss = [par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) for h in self.h]

        self.spike_loss = tf.reduce_mean(tf.stack(spike_loss, axis=0))

        #self.loss = self.pol_loss + self.val_loss + self.spike_loss
        self.stacked_mask = tf.stack(self.mask)

        adam_opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])

        self.train_opt = adam_opt.minimize(self.pol_loss + self.val_loss + self.spike_loss - 0.1*self.entropy_loss)



def main(gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    """
    Reset TensorFlow before running anything
    """
    tf.reset_default_graph()

    """
    Create the stimulus class to generate trial paramaters and input activity
    """
    stim = task.Stimulus()

    """
    Define all placeholder
    """
    mask = tf.placeholder(tf.float32, shape=[par['n_time_steps'], par['batch_size']])
    x = tf.placeholder(tf.float32, shape=[par['n_time_steps'], par['batch_size'], 32, 32, 3])  # input data
    target = tf.placeholder(tf.float32, shape=[par['n_time_steps'], par['batch_size'], par['n_pol']])  # input data
    actual_reward = tf.placeholder(tf.float32, shape=[par['n_time_steps'],par['n_val'],par['batch_size']])
    pred_reward = tf.placeholder(tf.float32, shape=[par['n_time_steps'], par['n_val'], par['batch_size']])
    actual_action = tf.placeholder(tf.float32, shape=[par['n_time_steps'], par['n_pol'], par['batch_size'], ])

    config = tf.ConfigProto()
    #config.gpu_options.allow_growth=True

    # enter "config=tf.ConfigProto(log_device_placement=True)" inside Session to check whether CPU/GPU in use
    with tf.Session(config=config) as sess:

        if gpu_id is not None:
            model = Model(x, target, actual_reward, pred_reward, actual_action, mask)
        else:
            with tf.device("/gpu:0"):
                model = Model(x, target, actual_reward, pred_reward, actual_action,mask)
        init = tf.global_variables_initializer()
        sess.run(init)

        # keep track of the model performance across training
        model_performance = {'accuracy': [], 'loss': [], 'perf_loss': [], 'spike_loss': [], 'trial': []}

        for i in range(par['num_iterations']):

            # generate batch of batch_train_size
            trial_info = stim.generate_trial()

            """
            Run the model
            """
            pol_out, val_out, pol_rnn, action, stacked_mask, reward = sess.run([model.pol_out, model.val_out, model.h_pol, model.action, \
                 model.stacked_mask,model.reward], {x: trial_info['neural_input'], target: trial_info['desired_output'], mask: trial_info['train_mask']})

            trial_reward = np.squeeze(np.stack(reward))
            trial_action = np.stack(action)
            #plt.imshow(np.squeeze(trial_reward))
            #plt.colorbar()
            #plt.show()

            _, pol_loss, val_loss = sess.run([model.train_opy, model.pol_loss, model.val_loss], \
                {x: trial_info['neural_input'], target: trial_info['desired_output'], mask: trial_info['train_mask'], \
                actual_reward: trial_reward, pred_reward: np.squeeze(val_out), actual_action:trial_action })


            accuracy, _, _ = analysis.get_perf(trial_info['desired_output'], action, trial_info['train_mask'])

            #model_performance = append_model_performance(model_performance, accuracy, val_loss, pol_loss, spike_loss, (i+1)*N)

            """
            Save the network model and output model performance to screen
            """
            if i%par['iters_between_outputs']==0 and i > 0:
                print_results(i, N, pol_loss, 0., pol_rnn, accuracy)
                r = np.squeeze(np.sum(np.stack(trial_reward),axis=0))
                print('Mean mask' , np.mean(stacked_mask), ' val loss ', val_loss, ' reward ', np.mean(r), np.max(r))
                #plt.imshow(np.squeeze(stacked_mask[:,:]))
                #plt.colorbar()
                #plt.show()
                #plt.imshow(np.squeeze(trial_reward))
                #plt.colorbar()
                #plt.show()


        """
        Save model, analyze the network model and save the results
        """
        #save_path = saver.save(sess, par['save_dir'] + par['ckpt_save_fn'])
        if par['analyze_model']:
            weights = eval_weights()
            analysis.analyze_model(trial_info, y_hat, state_hist, syn_x_hist, syn_u_hist, model_performance, weights, \
                simulation = True, lesion = False, tuning = False, decoding = False, load_previous_file = False, save_raw_data = False)

            # Generate another batch of trials with test_mode = True (sample and test stimuli
            # are independently drawn), and then perform tuning and decoding analysis
            trial_info = stim.generate_trial(test_mode = True)
            y_hat, state_hist, syn_x_hist, syn_u_hist = \
                sess.run([model.y_hat, model.hidden_state_hist, model.syn_x_hist, model.syn_u_hist], \
                {x: trial_info['neural_input'], y: trial_info['desired_output'], mask: trial_info['train_mask']})
            analysis.analyze_model(trial_info, y_hat, state_hist, syn_x_hist, syn_u_hist, model_performance, weights, \
                simulation = False, lesion = False, tuning = par['analyze_tuning'], decoding = True, load_previous_file = True, save_raw_data = False)



def append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, trial_num):

    model_performance['accuracy'].append(accuracy)
    model_performance['loss'].append(loss)
    model_performance['perf_loss'].append(perf_loss)
    model_performance['spike_loss'].append(spike_loss)
    model_performance['trial'].append(trial_num)

    return model_performance

def eval_weights():

    with tf.variable_scope('rnn_cell', reuse=True):
        W_in = tf.get_variable('W_in')
        W_rnn = tf.get_variable('W_rnn')
        b_rnn = tf.get_variable('b_rnn')

    with tf.variable_scope('output', reuse=True):
        W_out = tf.get_variable('W_out')
        b_out = tf.get_variable('b_out')

    weights = {
        'w_in'  : W_in.eval(),
        'w_rnn' : W_rnn.eval(),
        'w_out' : W_out.eval(),
        'b_rnn' : b_rnn.eval(),
        'b_out'  : b_out.eval()
    }

    return weights

def print_results(iter_num, trials_per_iter, perf_loss, spike_loss, state_hist, accuracy):

    print('Iter. {:4d}'.format(iter_num) + ' | Accuracy {:0.4f}'.format(accuracy) +
      ' | Perf loss {:0.4f}'.format(perf_loss) + ' | Spike loss {:0.4f}'.format(spike_loss) +
      ' | Mean activity {:0.4f}'.format(np.mean(state_hist)))
