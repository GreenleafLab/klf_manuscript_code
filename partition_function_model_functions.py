import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys  
sys.path.insert(0, '.')
import common
import importlib
importlib.reload(common)
import itertools
from scipy.optimize import minimize

# untouched from partition_function_v2
def convert_to_probabilities(energies, kt=1):
    '''
    Function to convert energies to probabilities using Boltzmann
    '''
    unscaled_probs = np.exp(-1*energies.astype(np.float64)/kt)
    return pd.DataFrame(unscaled_probs / unscaled_probs.sum(), index=energies.index, columns=['prob'])

# updated to weigh by percentage of nucleosomes not half over under
def assign_energy_two_param(mat, tf_e, nuc_e, enhancer_bounds):
    '''
    Function for assigning energies to each state in a data frame
    This is our "2 parameter" model with only TF and nucleoosme energies.
    '''
    num_tfs = mat.filter(like='tfbs_').sum(axis=1)
    num_nuc = pd.Series(0, index=mat.index)
    avg_nuc_len = 130
    
    promoter_threshold = enhancer_bounds[0]
    enhancer_threshold = enhancer_bounds[1]
    for idx_nuc in range(len(mat.filter(like='nuc').columns.tolist())//3):
        # iterate through each nucleosome, this is calculating the average percentage over the enhancer bounds
        tmp_nuc_num = pd.Series(0, index=mat.index)
        total_nuc = np.zeros(len(tmp_nuc_num))
        over_start = mat['nuc{}_start'.format(idx_nuc+1)] <= promoter_threshold
        over_end = mat['nuc{}_end'.format(idx_nuc+1)] > enhancer_threshold
        total_nuc += (mat['nuc{}_end'.format(idx_nuc+1)]-promoter_threshold)*over_start
        total_nuc += (enhancer_threshold - mat['nuc{}_start'.format(idx_nuc+1)])*over_end
        total_nuc += (mat['nuc{}_end'.format(idx_nuc+1)] - mat['nuc{}_start'.format(idx_nuc+1)])*(~over_start*~over_end)
        tmp_nuc_num += total_nuc 
        tmp_nuc_num = tmp_nuc_num.fillna(0)
        num_nuc += tmp_nuc_num.copy()
    num_nuc = num_nuc/avg_nuc_len
        
    return num_tfs * tf_e + num_nuc * nuc_e

def assign_energy_three_param(mat, tf_e, nuc_e, delta_e, enhancer_bounds):
    '''
    Function for assigning energies to each state in a data frame
    This is our "3 parameter" model, which has a TF-dependent nucleosome energy
    Here, "delta_e" is a discounting factor on the nuc energy
    '''
    num_tfs = mat.filter(like='tfbs_').sum(axis=1)
    num_nuc = pd.Series(0, index=mat.index)
    avg_nuc_len = 130
    
    promoter_threshold = enhancer_bounds[0]
    enhancer_threshold = enhancer_bounds[1]
    for idx_nuc in range(len(mat.filter(like='nuc').columns.tolist())//3):
        # iterate through each nucleosome, this is calculating the average percentage over the enhancer bounds
        tmp_nuc_num = pd.Series(0, index=mat.index)        
        total_nuc = np.zeros(len(tmp_nuc_num))
        over_start = mat['nuc{}_start'.format(idx_nuc+1)] <= promoter_threshold
        over_end = mat['nuc{}_end'.format(idx_nuc+1)] > enhancer_threshold
        total_nuc += (mat['nuc{}_end'.format(idx_nuc+1)]-promoter_threshold)*over_start
        total_nuc += (enhancer_threshold - mat['nuc{}_start'.format(idx_nuc+1)])*over_end
        total_nuc += (mat['nuc{}_end'.format(idx_nuc+1)] - mat['nuc{}_start'.format(idx_nuc+1)])*(~over_start*~over_end)
        tmp_nuc_num += total_nuc 
        tmp_nuc_num = tmp_nuc_num.fillna(0)
        num_nuc += tmp_nuc_num.copy()
    num_nuc = num_nuc/avg_nuc_len
        
    return num_tfs * tf_e + num_nuc * (nuc_e - delta_e * (num_tfs>0))

def get_summary_stats(states_list_, tf_e, delta_e=0, nuc_e = -1.2):
    '''
    Function for generating simulate TF and nucleosome occupancy summary statistics from sates list and defined energies
    '''
    simulated_data_3_param = []

    n_samples = 10000

    for idx in range(len(states_list_)):
        states = states_list_[idx]
        probs = convert_to_probabilities(assign_energy_three_param(states, tf_e, nuc_e, delta_e, (100,600)))        
        samp = np.random.choice(states['idx'], p=probs['prob'].values, size=n_samples)
        simulated_data_3_param.append(samp)

    simulated_avgTF_3_param = []
    for idx in range(len(states_list_)):
        states = states_list_[idx]
        simulated_3_param = (states.loc[simulated_data_3_param[idx]].filter(like='tfbs_').sum(axis=1)).mean()
        simulated_avgTF_3_param.append(simulated_3_param)
        
    promoter_threshold = 280
    enhancer_threshold = 430

    nuc_occupancy = []
    percent_nuc_samp = []
    for idx in range(len(states_list_)):
        states = states_list_[idx]
        states = states.loc[simulated_data_3_param[idx]].reset_index()
        n_nucs = len(states.filter(like='present').columns)
        nucs = np.zeros((len(states), 750))
        percent_nuc = []
        for idx2 in range(len(states)):
            mol = states.loc[idx2]
            total_nuc = 0
            
            for n in range(1, n_nucs+1):
                nuc = 'nuc{}'.format(n)
                if mol['nuc{}_present'.format(n)] == True:
                    
                    for p in range(int(mol['nuc{}_start'.format(n)]), int(mol['nuc{}_end'.format(n)])):
                        nucs[idx2,p] = 1
                    if mol[nuc+'_end'] > promoter_threshold:
                        if mol[nuc+'_start'] <= promoter_threshold:
                            total_nuc += mol[nuc+'_end']-promoter_threshold
                        elif mol[nuc+'_end'] > enhancer_threshold:
                            total_nuc += enhancer_threshold - mol[nuc+'_start']
                        else:
                            total_nuc += mol[nuc+'_end'] - mol[nuc+'_start']
            percent_nuc.append(total_nuc/(enhancer_threshold-promoter_threshold))
            
        percent_nuc_samp.append(np.average(percent_nuc))
        nuc_occupancy.append(pd.DataFrame(nucs).mean()[250:440])
    
    return simulated_avgTF_3_param, nuc_occupancy, percent_nuc_samp


def min_func(x, nuc_occupancy_ref, states_list):
    '''
    Function to pass in to scipy.minimize with setting 'Nelder-Mead' to fit partition function model on
    average nucleosome occupancy across each construct
    '''
    tf_e = x[0]
    delta_e = x[1]
    simulated_avgTF_3_param, nuc_occupancy, percent_nuc_samp = get_summary_stats(states_list, tf_e, delta_e)
    nuc_occ_all_flat = list(itertools.chain(*nuc_occupancy))
    nuc_r2 = r2_score(nuc_occupancy_ref, nuc_occ_all_flat)
    mse = np.mean((np.asarray(nuc_occupancy_ref) - np.asarray(nuc_occ_all_flat)) ** 2)
    return mse