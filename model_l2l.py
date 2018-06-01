import tensorflow as tf
import numpy as np
import task
import matplotlib.pyplot as plt
from parameters import par
from convolutional_layers import apply_convolutional_layers
import os, sys

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print('TensorFlow version:\t', tf.__version__)
print('Using EI Network:\t', par['EI'])
print('Synaptic configuration:\t', par['synapse_config'], "\n")

# cell state placeholder for vanilla RNN
invalid_c_val = 0

"""
Model setup and execution
"""
class Model:

    def __init__(self, input_data, target_data, pred_val, actual_action, advantage, mask, new_trial, h_init, c_init):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=0)
        self.target_data = tf.unstack(target_data, axis=0)
        self.pred_val = tf.unstack(pred_val, axis=0)
        self.actual_action = tf.unstack(actual_action, axis=0)
        self.advantage = tf.unstack(advantage, axis=0)
        self.new_trial = tf.unstack(new_trial)
        self.W_ei = tf.constant(par['EI_matrix'])

        self.time_mask = tf.unstack(mask, axis=0)

        # Load the initial hidden state activity to be used at the start of each trial
        self.h_init = h_init
        self.c_init = c_init

        # Build the TensorFlow graph
        self.rnn_cell_loop(self.h_init, self.c_init)

        # Train the model
        self.optimize()


    def rnn_cell_loop(self, h, c):


        self.W_ei = tf.constant(par['EI_matrix'])
        self.h = [] # RNN activity
        self.c = [] # RNN activity
        self.pol_out = [] # policy output
        self.val_out = [] # value output
        self.syn_x = [] # STP available neurotransmitter, currently not in use
        self.syn_u = [] # STP calcium concentration, currently not in use

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
        self.define_vars()


        """
        Loop through the neural inputs to the RNN, indexed in time
        """
        for rnn_input, target, time_mask, new_trial in zip(self.input_data, self.target_data, self.time_mask, self.new_trial):

            x = apply_convolutional_layers(rnn_input, par['conv_weight_fn'])
            self.conv_output = tf.transpose(x)

            h, c, action, pol_out, val_out, mask, reward  = self.rnn_cell(self.conv_output, h, c, self.action[-1], self.reward[-1], \
                self.mask[-1], target, time_mask, new_trial)

            self.h.append(h)
            self.c.append(c)
            self.action.append(tf.transpose(action))
            self.pol_out.append(pol_out)
            self.val_out.append(val_out)
            self.mask.append(mask)
            self.reward.append(tf.reshape(reward, [par['n_val'], par['batch_size']]))

        self.mask = self.mask[1:]
        # actions will produce a reward on the next time step
        self.reward = self.reward[1:]
        self.action = self.action[1:]


    def rnn_cell(self, x, h, c, prev_action, prev_reward, mask, target, time_mask, new_trial):

        # Modify the recurrent weights if using excitatory/inhibitory neurons
        if par['EI']:
            if par['LSTM']:
                self.Uf = tf.matmul(tf.nn.relu(self.Uf), self.W_ei)
                self.Ui = tf.matmul(tf.nn.relu(self.Ui), self.W_ei)
                self.Uo = tf.matmul(tf.nn.relu(self.Uo), self.W_ei)
                self.Uc = tf.matmul(tf.nn.relu(self.Uc), self.W_ei)

            else:
                self.W_rnn = tf.matmul(tf.nn.relu(self.W_rnn), self.W_ei)

        # pass the output of the convolutional layers through the feedforward layer(s)
        if par['include_ff_layer']:
            x = tf.nn.relu(tf.matmul(self.W_in0, x) + self.b_in0)

        [h, c] = self.recurrent_cell(h, c, x, prev_action, prev_reward, mask)

        # calculate the policy output and choose an action
        pol_out = tf.matmul(self.W_pol_out, h) + self.b_pol_out
        action_index = tf.multinomial(tf.transpose(pol_out), 1)
        action = tf.one_hot(tf.squeeze(action_index), par['n_pol'])
        action = tf.reshape(action, [par['batch_size'], par['n_pol']])
        pol_out = tf.nn.softmax(pol_out, dim = 0) # needed for optimize

        val_out = tf.matmul(self.W_val_out, h) + self.b_val_out

        # if previous reward was non-zero, then end the trial, unless the new trial signal cue is on
        continue_trial = tf.cast(tf.equal(prev_reward, 0.), tf.float32)
        mask *= continue_trial
        mask = tf.maximum(new_trial, mask)
        continue_trial = tf.maximum(new_trial, continue_trial)

        reward = tf.reduce_sum(action*target, axis = 1)*mask*time_mask

        return h, c, action, pol_out, val_out, mask, reward


    def optimize(self):

        epsilon = 1e-7
        var_list = [var for var in tf.trainable_variables()]
        #Z = tf.reduce_sum(tf.stack([tf.reduce_sum(time_mask*mask) for (mask, time_mask) in zip(self.mask, self.time_mask)]))

        self.pol_loss = -tf.reduce_sum(tf.stack([advantage*time_mask*mask*tf.reduce_sum(act*tf.log((epsilon + pol_out)), axis = 0) \
            for (pol_out, advantage, act, mask, time_mask) in zip(self.pol_out, self.advantage, \
            self.actual_action, self.mask, self.time_mask)]))

        self.entropy_loss = -par['entropy_cost']*tf.reduce_sum(tf.stack([time_mask*mask*pol_out*tf.log(epsilon+pol_out) \
            for (pol_out, mask, time_mask) in zip(self.pol_out, self.mask, self.time_mask)]))

        self.val_loss = 0.5*tf.reduce_sum(tf.stack([time_mask*mask*tf.square(val_out - pred_val) \
            for (val_out, mask, time_mask, pred_val) in zip(self.val_out[:-1], self.mask, self.time_mask, self.pred_val[1:])]))

        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        self.spike_loss = tf.reduce_mean(tf.stack([par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) for h in self.h]))

        adam_opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])

        """
        Calculate gradients and add accumulate
        """
        self.cummulative_grads = {}
        update_gradients = []
        reset_gradients = []
        for var in var_list:
            self.cummulative_grads[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable = False)
        grads_and_vars = adam_opt.compute_gradients(self.pol_loss + self.val_loss + self.spike_loss - self.entropy_loss)
        for grad, var in grads_and_vars:
            #grad = tf.clip_by_norm(grad, par['clip_max_grad_val'])
            update_gradients.append(tf.assign_add(self.cummulative_grads[var.op.name], grad))
            reset_gradients.append(tf.assign(self.cummulative_grads[var.op.name], 0.*self.cummulative_grads[var.op.name]))

        #with tf.control_dependencies([update_gradients]):
        self.update_gradients = tf.group(*update_gradients)
        self.reset_gradients = tf.group(*reset_gradients)

        """
        Apply gradients
        """
        capped_gvs = []
        for var in var_list:
            #capped_gvs.append((self.cummulative_grads[var.op.name], var))
            if var.name == "recurrent_pol/W_rnn:0":
                self.cummulative_grads[var.op.name] *= par['w_rnn_mask']
            capped_gvs.append((tf.clip_by_norm(self.cummulative_grads[var.op.name], par['clip_max_grad_val']), var))
        self.train_opt = adam_opt.apply_gradients(capped_gvs)


    def recurrent_cell(self, h, c, x, prev_action, prev_reward, mask):

        if par['LSTM']:
            # forgetting gate
            f = tf.sigmoid(tf.matmul(self.Wf, x) + tf.matmul(self.Uf, h) + tf.matmul(self.Wf_reward_pos, tf.nn.relu(prev_reward)) \
                + tf.matmul(self.Wf_reward_neg, tf.nn.relu(-prev_reward)) + mask*(tf.matmul(self.Wf_action, prev_action)) + self.bf)
            # input gate
            i = tf.sigmoid(tf.matmul(self.Wi, x) + tf.matmul(self.Ui, h) + tf.matmul(self.Wi_reward_pos, tf.nn.relu(prev_reward)) \
                + tf.matmul(self.Wi_reward_neg, tf.nn.relu(-prev_reward)) + mask*(tf.matmul(self.Wi_action, prev_action)) + self.bi)
            # updated cell state
            cn = tf.tanh(tf.matmul(self.Wc, x) + tf.matmul(self.Uc, h) + tf.matmul(self.Wc_reward_pos, tf.nn.relu(prev_reward)) \
                + tf.matmul(self.Wc_reward_neg, tf.nn.relu(-prev_reward)) + mask*(tf.matmul(self.Wc_action, prev_action)) + self.bc)
            c = tf.multiply(f, c) + tf.multiply(i, cn)
            # output gate
            o = tf.sigmoid(tf.matmul(self.Wo, x) + tf.matmul(self.Uo, h) + tf.matmul(self.Wo_reward_pos, tf.nn.relu(prev_reward)) \
                + tf.matmul(self.Wo_reward_neg, tf.nn.relu(-prev_reward)) + mask*(tf.matmul(self.Wo_action, prev_action)) + self.bo)

            h = tf.multiply(o, tf.tanh(c))

        else:
            h = tf.nn.relu(h*(1-par['alpha']) + par['alpha']*(tf.matmul(self.W_in1, x) + tf.matmul(self.W_rnn, h) \
                + mask*(tf.matmul(self.W_reward_pos, tf.nn.relu(prev_reward)) + tf.matmul(self.W_reward_neg, tf.nn.relu(-prev_reward)) \
                + tf.matmul(self.W_action, prev_action)) + self.b_rnn + tf.random_normal([par['n_hidden'], par['batch_size']], 0, par['noise_rnn'], dtype=tf.float32)))

            c = invalid_c_val

        return h, c


    def define_vars(self):

        # in TF v1.8, I can use reuse = tf.AUTO_REUSE, and get rid of first weight initialization above

        # W_in0, and W_in1 are feedforward weights whose input is the convolved image, and projects onto the RNN
        # W_reward_pos, W_reward_neg project the postive and negative part of the reward from the previous time point onto the RNN
        # W_action projects the action from the previous time point onto the RNN
        # Wnn projects the activity of the RNN from the previous time point back onto the RNN (i.e. the recurrent weights)
        # W_pol_out projects from the RNN onto the policy output neurons
        # W_val_out projects from the RNN onto the value output neuron

        with tf.variable_scope('recurrent_pol'):
            if par['include_ff_layer']:
                self.W_in0 = tf.get_variable('W_in0', initializer = par['W_in0_init'])
                self.b_in0 = tf.get_variable('b_in0', initializer = par['b_in0_init'])

            self.W_pol_out = tf.get_variable('W_pol_out', initializer = par['W_pol_out_init'])
            self.b_pol_out = tf.get_variable('b_pol_out', initializer = par['b_pol_out_init'])

            self.W_val_out = tf.get_variable('W_val_out', initializer = par['W_val_out_init'])
            self.b_val_out = tf.get_variable('b_val_out', initializer = par['b_val_out_init'])

        if par['LSTM']:
            # following conventions on https://en.wikipedia.org/wiki/Long_short-term_memory
            self.Wf = tf.get_variable('Wf', initializer = par['Wf_init'])
            self.Wi = tf.get_variable('Wi', initializer = par['Wi_init'])
            self.Wo = tf.get_variable('Wo', initializer = par['Wo_init'])
            self.Wc = tf.get_variable('Wc', initializer = par['Wc_init'])

            self.Uf = tf.get_variable('Uf', initializer = par['Uf_init'])
            self.Ui = tf.get_variable('Ui', initializer = par['Ui_init'])
            self.Uo = tf.get_variable('Uo', initializer = par['Uo_init'])
            self.Uc = tf.get_variable('Uc', initializer = par['Uc_init'])

            self.bf = tf.get_variable('bf', initializer = par['bf_init'])
            self.bi = tf.get_variable('bi', initializer = par['bi_init'])
            self.bo = tf.get_variable('bo', initializer = par['bo_init'])
            self.bc = tf.get_variable('bc', initializer = par['bc_init'])

            self.Wf_reward_pos = tf.get_variable('Wf_reward_pos', initializer = par['Wf_reward_pos_init'])
            self.Wi_reward_pos = tf.get_variable('Wi_reward_pos', initializer = par['Wi_reward_pos_init'])
            self.Wo_reward_pos = tf.get_variable('Wo_reward_pos', initializer = par['Wo_reward_pos_init'])
            self.Wc_reward_pos = tf.get_variable('Wc_reward_pos', initializer = par['Wc_reward_pos_init'])

            self.Wf_reward_neg = tf.get_variable('Wf_reward_neg', initializer = par['Wf_reward_neg_init'])
            self.Wi_reward_neg = tf.get_variable('Wi_reward_neg', initializer = par['Wi_reward_neg_init'])
            self.Wo_reward_neg = tf.get_variable('Wo_reward_neg', initializer = par['Wo_reward_neg_init'])
            self.Wc_reward_neg = tf.get_variable('Wc_reward_neg', initializer = par['Wc_reward_neg_init'])

            self.Wf_action = tf.get_variable('Wf_action', initializer = par['Wf_action_init'])
            self.Wi_action = tf.get_variable('Wi_action', initializer = par['Wi_action_init'])
            self.Wo_action = tf.get_variable('Wo_action', initializer = par['Wo_action_init'])
            self.Wc_action = tf.get_variable('Wc_action', initializer = par['Wc_action_init'])

        else:
            # Weights for vanilla RNN
            self.W_in1 = tf.get_variable('W_in1', initializer = par['W_in1_init'])
            self.b_rnn = tf.get_variable('b_rnn', initializer = par['b_rnn_init'])
            self.W_rnn = tf.get_variable('W_rnn', initializer = par['W_rnn'])
            self.W_reward_pos = tf.get_variable('W_reward_pos', initializer = par['W_reward_pos_init'])
            self.W_reward_neg = tf.get_variable('W_reward_neg', initializer = par['W_reward_neg_init'])
            self.W_action = tf.get_variable('W_action', initializer = par['W_action_init'])

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
    x, target, mask, pred_val, actual_action, advantage, new_trial, h_init, c_init, mask = generate_placeholders()

    config = tf.ConfigProto()
    #config.gpu_options.allow_growth=True

    with tf.Session(config = config) as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, target, pred_val, actual_action, advantage, mask, new_trial, h_init, c_init)

        sess.run(tf.global_variables_initializer())

        # keep track of the model performance across training
        model_performance = {'reward': [], 'entropy_loss': [], 'val_loss': [], 'pol_loss': [], 'spike_loss': [], 'trial': []}

        hidden_init = np.array(par['h_init'])
        cell_state_init = np.array(par['c_init'])

        for i in range(par['num_iterations']):

            """
            Generate stimulus and response contigencies
            """
            input_data, reward_data, trial_mask, new_trial_signal = stim.generate_batch_task1(0)

            """
            Run the model
            """
            pol_out_list, val_out_list, h_list, action_list, mask_list, reward_list = sess.run([model.pol_out, model.val_out, model.h, model.action, \
                 model.mask, model.reward], {x: input_data, target: reward_data, mask: trial_mask, new_trial: new_trial_signal, h_init:hidden_init, c_init:cell_state_init})

            """
            Unpack all lists, calculate predicted value and advantage functions
            """
            val_out, reward, adv, act, predicted_val, stacked_mask = stack_vars(pol_out_list, val_out_list, reward_list, action_list, mask_list, trial_mask)

            """
            Calculate and accumulate gradients
            """
            _, pol_loss, val_loss, entropy_loss = sess.run([model.update_gradients, model.pol_loss, model.val_loss, model.entropy_loss], \
                {x: input_data, target: reward_data, mask: trial_mask, pred_val: predicted_val, \
                actual_action: act, advantage:adv, new_trial: new_trial_signal, h_init: hidden_init, c_init: cell_state_init})

            """
            cg = sess.run([model.cummulative_grads])
            c = 0
            for k,v in cg[0].items():
                c+=np.sum(v**2)
            print(i,c)
            """

            """
            Apply the accumulated gradients and reset
            """
            if i>0 and i%par['trials_per_grad_update'] == 0:
                sess.run([model.train_opt])
                sess.run([model.reset_gradients])

            hidden_init = np.array(h_list[-1])

            """
            Append model results an dprint results
            """
            append_model_performance(model_performance, reward, entropy_loss, pol_loss, val_loss, i)
            if i%par['iters_between_outputs']==0 and i > 0:
                print_results(i, model_performance)



def stack_vars(pol_out_list, val_out_list, reward_list, action_list, mask_list, trial_mask):


    pol_out = np.stack(pol_out_list)
    val_out = np.stack(val_out_list)
    stacked_mask = np.stack(mask_list)[:,0,:]*trial_mask
    reward = np.stack(reward_list)
    val_out_stacked = np.vstack((np.zeros((1,par['n_val'],par['batch_size'])), val_out))
    terminal_state = np.float32(reward != 0) # this assumes that the trial ends when a reward other than zero is received
    pred_val = reward + par['discount_rate']*val_out_stacked[1:,:,:]*(1-terminal_state)
    adv = pred_val - val_out_stacked[:-1,:,:]
    #adv = reward - val_out
    act = np.stack(action_list)

    return val_out, reward, adv, act, pred_val, stacked_mask

def append_model_performance(model_performance, reward, entropy_loss, pol_loss, val_loss, trial_num):

    reward = np.mean(np.sum(reward,axis = 0))/par['trials_per_sequence']
    model_performance['reward'].append(reward)
    model_performance['entropy_loss'].append(entropy_loss)
    model_performance['pol_loss'].append(pol_loss)
    model_performance['val_loss'].append(val_loss)
    model_performance['trial'].append(trial_num)

    return model_performance

def generate_placeholders():

    mask = tf.placeholder(tf.float32, shape=[par['sequence_time_steps'], par['batch_size']])
    x = tf.placeholder(tf.float32, shape=[par['sequence_time_steps'], par['batch_size'], 32, 32, 3])  # input data
    target = tf.placeholder(tf.float32, shape=[par['sequence_time_steps'], par['batch_size'], par['n_pol']])  # input data
    pred_val = tf.placeholder(tf.float32, shape=[par['sequence_time_steps'], par['n_val'], par['batch_size']])
    actual_action = tf.placeholder(tf.float32, shape=[par['sequence_time_steps'], par['n_pol'], par['batch_size']])
    advantage  = tf.placeholder(tf.float32, shape=[par['sequence_time_steps'], par['n_val'], par['batch_size']])
    new_trial  = tf.placeholder(tf.float32, shape=[par['sequence_time_steps']])
    h_init =  tf.placeholder(tf.float32, shape=[par['n_hidden'],par['batch_size']])
    c_init =  tf.placeholder(tf.float32, shape=[par['n_hidden'],par['batch_size']])

    return x, target, mask, pred_val, actual_action, advantage, new_trial, h_init, c_init, mask

def eval_weights():

    # TODO: NEEDS FIXING!
    with tf.variable_scope('rnn_cell'):
        W_in = tf.get_variable('W_in')
        W_rnn = tf.get_variable('W_rnn')
        b_rnn = tf.get_variable('b_rnn')

    with tf.variable_scope('output'):
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

def print_results(iter_num, model_performance):

    reward = np.mean(np.stack(model_performance['reward'])[-par['iters_between_outputs']:])
    pol_loss = np.mean(np.stack(model_performance['pol_loss'])[-par['iters_between_outputs']:])
    val_loss = np.mean(np.stack(model_performance['val_loss'])[-par['iters_between_outputs']:])
    entropy_loss = np.mean(np.stack(model_performance['entropy_loss'])[-par['iters_between_outputs']:])

    print('Iter. {:4d}'.format(iter_num) + ' | Reward {:0.4f}'.format(reward) +
      ' | Pol loss {:0.4f}'.format(pol_loss) + ' | Val loss {:0.4f}'.format(val_loss) +
      ' | Entropy loss {:0.4f}'.format(entropy_loss))
