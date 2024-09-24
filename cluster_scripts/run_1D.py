#!/usr/bin/env python3

import numpy as np
from tenpy.networks.mps import MPS
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.algorithms import dmrg
import pickle, time

out_dir = "/home/tenkila2/scratch/DMRG_out/"
#out_dir = " ./"#"/home/gaurav/Projects/DMRG_HK/output/"

st_time_full = time.time()
# Parameters
U_list = np.array([0.1, 0.2, 0.5, 1, 2, 4, 6, 8])
norb_list = np.array([40])
#t = 1.0
#mu = 0  # Chemical potential (often set to 0 for simplicity at half filling)
chi_max = 1000  # Maximum bond dimension
sweeps = 100  # Number of DMRG sweeps
k_num = 6

twist_avg_energy =  np.zeros(shape=(len(U_list), len(norb_list)))
twist_avg_gse =  np.zeros(shape=(len(U_list), len(norb_list)))
time_list = np.zeros(shape=(len(U_list), len(norb_list)))

         
def run_dmrg_half_filling(L, U, t, mu, chi_max, sweeps, hc_flag=False, charge=0):
    model_params = {
        'L': L,
        'U': U,
        't':t,
        'mu': mu,
        'bc_MPS': 'finite',
        'bc_x':'periodic',
        'explicit_plus_hc':hc_flag
    }

    model = FermiHubbardModel(model_params)

    # Initialize the MPS for half-filling with an alternating pattern of up and down spins
    if charge==0:
        initial_state = ['down', 'up'] * (L // 2) if L % 2 == 0 else ['down', 'up'] * (L // 2) + ['down']
    elif charge==1:
        initial_state = ['down', 'up'] * (L // 2) if L % 2 == 0 else ['down', 'up'] * (L // 2) + ['full']
        if L%2==0: initial_state[-1] = 'full'
    elif charge==-1:
        initial_state = ['down', 'up'] * (L // 2) if L % 2 == 0 else ['down', 'up'] * (L // 2) + ['empty']
        if L%2==0: initial_state[-1] = 'empty'
    psi = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc='finite')

    dmrg_params = {
        'mixer': True,  # Enables mixing for better convergence in difficult phases
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': 1.e-6
        },
        'max_sweeps': sweeps,
    }

    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E0, psi = eng.run()

    return E0


for i, U in enumerate(U_list):
    for j, norb in enumerate(norb_list):
        mu = U/2
        st_time = time.time()
        energy_arr = np.zeros(k_num)
        energy_arr_gse = np.zeros(k_num)
        for kx in range(-int(k_num/2), int(k_num/2)):
            t_star = np.exp(1j*kx/k_num*np.pi)
            gse = run_dmrg_half_filling(norb, U, t_star, mu, chi_max, sweeps,charge=0)
            gap_energy = (2*run_dmrg_half_filling(norb, U, t_star, mu, chi_max, sweeps,charge=1) -2*gse)
            energy_arr[kx+int(k_num/2)] = gap_energy
            energy_arr_gse[kx+int(k_num/2)] = gse
        twist_avg_energy[i, j] = energy_arr.mean()
        twist_avg_gse[i, j] = gse
        time_list[i, j] = time.time() - st_time
        print(U, norb, time_list[i,j])
        print(twist_avg_energy)

data_dict = {}
data_dict['twist_avg_list'] = twist_avg_energy
data_dict['twist_avg_gse'] = twist_avg_gse
data_dict['U_list'] = U_list
data_dict['norb_list'] = norb_list
data_dict['chi_max'] = chi_max
data_dict['sweeps'] = sweeps
data_dict['duration'] = time.time() - st_time_full
data_dict['k_num'] = k_num
data_dict['time_list'] = time_list
print(data_dict['duration'])
#Give file a unique name with timestamp and unique serial number

filename = '1D_OHK_DMRG_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl'
with open(out_dir+filename, 'wb') as f:
    pickle.dump(data_dict, f)

