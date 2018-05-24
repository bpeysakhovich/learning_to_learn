import numpy as np
import tensorflow as tf
import os

print("--> Loading parameters...")

"""
Independent parameters
"""

par = {
    # Setup parameters
    'save_dir'              : './savedir/',
    'conv_weight_fn'        : '/home/masse/Context-Dependent-Gating/savedir/conv_weights_test.pkl',
    'analyze_model'         : True,

    # Network configuration
    'synapse_config'        : None, # Full is 'std_stf'
    'exc_inh_prop'          : 0.8,       # Literature 0.8, for EI off 1
    'var_delay'             : False,

    # Network shape
    'n_input'               : [2048, 1000],
    'n_hidden'              : 100,
    'n_pol'                 : 3,
    'n_val'                 : 1,
    'include_ff_layer'      : False,

    # Timings and rates
    'dt'                    : 20,
    'learning_rate'         : 1e-3,
    'membrane_time_constant': 100,
    'connection_prob'       : 1,         # Usually 1


    # Variance values
    'clip_max_grad_val'     : 1,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.05,
    'noise_rnn_sd'          : 0.05,
    'drop_keep_pct'         : 0.8,

    # Tuning function data
    'num_motion_dirs'       : 8,
    'tuning_height'         : 4,        # magnitutde scaling factor for von Mises
    'kappa'                 : 2,        # concentration scaling factor for von Mises

    # Cost parameters
    'spike_cost'            : 1e-6,
    'wiring_cost'           : 0.,

    # Synaptic plasticity specs
    'tau_fast'              : 200,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_size'            : 8,
    'num_iterations'        : 20000,
    'iters_between_outputs' : 10,
    'trials_per_sequence'   : 2,

    # Task specs
    'trial_type'            : 'DMS', # allowable types: DMS, DMRS45, DMRS90, DMRS180, DMC, DMS+DMRS, ABBA, ABCA, dualDMS
    'rotation_match'        : 0,  # angular difference between matching sample and test
    'dead_time'             : 200,
    'fix_time'              : 200,
    'sample_time'           : 400,
    'delay_time'            : 100,
    'test_time'             : 400,
    'variable_delay_max'    : 300,
    'mask_duration'         : 50,  # duration of traing mask after test onset
    'catch_trial_pct'       : 0.0,
    'num_receptive_fields'  : 1,
    'num_rules'             : 1, # this will be two for the DMS+DMRS task

    # Save paths
    'save_fn'               : 'model_results.pkl',

    # Analysis
    'svm_normalize'         : True,
    'decoding_reps'         : 100,
    'simulation_reps'       : 100,
    'decode_test'           : False,
    'decode_rule'           : True,
    'decode_sample_vs_test' : False,
    'suppress_analysis'     : False,
    'analyze_tuning'        : False,

}


"""
Dependent parameters
"""

def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    print('Updating parameters...')
    for key, val in updates.items():
        par[key] = val
        #print('Updating ', key)

    update_trial_params()
    update_dependencies()

def update_trial_params():

    """
    Update all the trial parameters given trial_type
    """

    par['num_rules'] = 1
    par['num_rule_tuned'] = 0
    par['ABBA_delay' ] = 0

    if par['trial_type'] == 'DMS' or par['trial_type'] == 'DMC':
        par['rotation_match'] = 0

    elif par['trial_type'] == 'DMRS45':
        par['rotation_match'] = 45

    elif par['trial_type'] == 'DMRS90':
        par['rotation_match'] = 90

    elif par['trial_type'] == 'DMRS90ccw':
        par['rotation_match'] = -90

    elif  par['trial_type'] == 'DMRS180':
        par['rotation_match'] = 180

    elif par['trial_type'] == 'dualDMS':
        par['catch_trial_pct'] = 0
        par['num_receptive_fields'] = 2
        par['num_rules'] = 2
        par['probe_trial_pct'] = 0
        par['probe_time'] = 10
        par['num_rule_tuned'] = 12
        par['sample_time'] = 500
        par['test_time'] = 500
        par['delay_time'] = 1000
        par['analyze_rule'] = True
        par['num_motion_tuned'] = 36
        par['noise_in_sd']  = 0.1
        par['noise_rnn_sd'] = 0.5
        par['num_iterations'] = 4000

        par['dualDMS_single_test'] = False

    elif par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA':
        par['catch_trial_pct'] = 0
        par['match_test_prob'] = 0.5
        par['max_num_tests'] = 3
        par['sample_time'] = 400
        par['delay_time'] = 2400
        par['ABBA_delay'] = par['delay_time']//par['max_num_tests']//2
        par['repeat_pct'] = 0
        par['analyze_test'] = True
        if par['trial_type'] == 'ABBA':
            par['repeat_pct'] = 0.5

    elif par['trial_type'] == 'DMS+DMRS' or par['trial_type'] == 'DMS+DMRS_early_cue':

        par['num_rules'] = 2
        par['num_rule_tuned'] = 12
        if par['trial_type'] == 'DMS+DMRS':
            par['rotation_match'] = [0, 90]
            par['rule_onset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + 500
            par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + 750
        else:
            par['rotation_match'] = [0, 45]
            par['rule_onset_time'] = par['dead_time']
            par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']-200

    elif par['trial_type'] == 'DMS+DMC':
        par['num_rules'] = 2
        par['num_rule_tuned'] = 12
        par['rotation_match'] = [0, 0]
        par['rule_onset_time'] = 0
        par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']

    elif par['trial_type'] == 'DMS+DMRS+DMC':
        par['num_rules'] = 3
        par['num_rule_tuned'] = 18
        par['rotation_match'] = [0, 90, 0]
        par['rule_onset_time'] = par['dead_time']
        par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']

    else:
        print(par['trial_type'], ' not a recognized trial type')
        quit()


def update_dependencies():
    """
    Updates all parameter dependencies
    """


    # Possible rules based on rule type values
    #par['possible_rules'] = [par['num_receptive_fields'], par['num_categorizations']]

    # If num_inh_units is set > 0, then neurons can be either excitatory or
    # inihibitory; is num_inh_units = 0, then the weights projecting from
    # a single neuron can be a mixture of excitatory or inhibitory
    if par['exc_inh_prop'] < 1:
        par['EI'] = True
    else:
        par['EI']  = False

    par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
    par['num_inh_units'] = par['n_hidden'] - par['num_exc_units']

    par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
    par['EI_list'][-par['num_inh_units']:] = -1.

    par['drop_mask'] = np.ones((par['n_hidden'],par['n_hidden']), dtype=np.float32)
    ind_inh = np.where(par['EI_list']==-1)[0]
    par['drop_mask'][:, ind_inh] = 0.
    par['drop_mask'][ind_inh, :] = 0.

    par['EI_matrix'] = np.diag(par['EI_list'])

    # Membrane time constant of RNN neurons
    par['alpha'] = np.float32(par['dt'])/par['membrane_time_constant']
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par['noise_rnn'] = np.sqrt(2*par['alpha'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha'])*par['noise_in_sd'] # since term will be multiplied by par['alpha']


    # The time step in seconds
    par['dt_sec'] = par['dt']/1000
    # Length of each trial in ms
    if par['trial_type'] == 'dualDMS' and not par['dualDMS_single_test']:
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+2*par['delay_time']+2*par['test_time']
    else:
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time']
    # Length of each trial in time steps
    par['n_time_steps'] = par['trial_length']//par['dt']


    ####################################################################
    ### Setting up assorted intial weights, biases, and other values ###
    ####################################################################

    par['h_init'] = 0.1*np.ones((par['n_hidden'], par['batch_size']), dtype=np.float32)

    # Initialize input weights
    c = 0.02
    if par['include_ff_layer']:
        par['W_in0_init'] =  c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_input'][1], par['n_input'][0]]))
        par['b_in0_init'] = np.zeros((par['n_input'][1], 1), dtype = np.float32)
        par['W_in1_init'] =  c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_hidden'], par['n_input'][1]]))

    else:
        #par['W_in1_init'] =  c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_hidden'], par['n_input'][0]]))
        par['W_in1_init'] =  c*np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_input'][0]]))

    if par['EI']:
        par['W_rnn_pol_init'] =  c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_hidden'], par['n_hidden']]))
        par['W_rnn_val_init'] =  c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_hidden'], par['n_hidden']]))
        par['w_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32) - np.eye(par['n_hidden'])
        par['W_rnn_pol_init'] *= par['w_rnn_mask']
        par['W_rnn_val_init'] *= par['w_rnn_mask']
    else:
        par['W_rnn_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
        par['w_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32)

    """
    par['W_reward_pos_init'] =  c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_hidden'], 1]))
    par['W_reward_neg_init'] =  c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_hidden'], 1]))
    par['W_action_init'] =  c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_hidden'], par['n_pol']]))
    par['W_pol_out_init'] =  np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_pol'], par['n_hidden']]))
    par['W_val_out_init'] =  np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_val'], par['n_hidden']]))
    """
    par['W_reward_pos_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], 1]))
    par['W_reward_neg_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], 1]))
    par['W_action_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_pol']]))
    par['W_pol_out_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_pol'], par['n_hidden']]))
    par['W_val_out_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_val'], par['n_hidden']]))

    par['b_rnn_init'] = np.zeros((par['n_hidden'], 1), dtype = np.float32)
    par['b_pol_out_init'] = np.zeros((par['n_pol'], 1), dtype = np.float32)
    par['b_val_out_init'] = np.zeros((par['n_val'], 1), dtype = np.float32)


    """
    Setting up synaptic parameters
    0 = static
    1 = facilitating
    2 = depressing
    """
    par['synapse_type'] = np.zeros(par['n_hidden'], dtype=np.int8)

    # only facilitating synapses
    if par['synapse_config'] == 'stf':
        par['synapse_type'] = np.ones(par['n_hidden'], dtype=np.int8)

    # only depressing synapses
    elif par['synapse_config'] == 'std':
        par['synapse_type'] = 2*np.ones(par['n_hidden'], dtype=np.int8)

    # even numbers facilitating, odd numbers depressing
    elif par['synapse_config'] == 'std_stf':
        par['synapse_tparaype'] = np.ones(par['n_hidden'], dtype=np.int8)
        par['ind'] = range(1,par['n_hidden'],2)
        par['synapse_type'][par['ind']] = 2

    par['alpha_stf'] = np.ones((par['n_hidden'], 1), dtype=np.float32)
    par['alpha_std'] = np.ones((par['n_hidden'], 1), dtype=np.float32)
    par['U'] = np.ones((par['n_hidden'], 1), dtype=np.float32)

    # initial synaptic values
    par['syn_x_init'] = np.zeros((par['n_hidden'], par['batch_size']), dtype=np.float32)
    par['syn_u_init'] = np.zeros((par['n_hidden'], par['batch_size']), dtype=np.float32)

    for i in range(par['n_hidden']):
        if par['synapse_type'][i] == 1:
            par['alpha_stf'][i,0] = par['dt']/par['tau_slow']
            par['alpha_std'][i,0] = par['dt']/par['tau_fast']
            par['U'][i,0] = 0.15
            par['syn_x_init'][i,:] = 1
            par['syn_u_init'][i,:] = par['U'][i,0]

        elif par['synapse_type'][i] == 2:
            par['alpha_stf'][i,0] = par['dt']/par['tau_fast']
            par['alpha_std'][i,0] = par['dt']/par['tau_slow']
            par['U'][i,0] = 0.45
            par['syn_x_init'][i,:] = 1
            par['syn_u_init'][i,:] = par['U'][i,0]

def initialize(dims, connection_prob):
    w = np.random.gamma(shape=0.25, scale=1.0, size=dims)
    #w = np.random.uniform(0,0.25, size=dims)
    w *= (np.random.rand(*dims) < connection_prob)
    return np.float32(w)


def spectral_radius(A):

    return np.max(abs(np.linalg.eigvals(A)))

update_trial_params()
update_dependencies()

print("--> Parameters successfully loaded.\n")
