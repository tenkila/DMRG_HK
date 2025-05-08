#!/usr/bin/env python3

import numpy as np
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.models.lattice import Square
from tenpy.algorithms import dmrg
from tenpy.models.model import CouplingModel, CouplingMPOModel
import pickle, time

#out_dir = "/home/tenkila2/scratch/DMRG_out/"
out_dir = "/home/gaurav/Projects/DMRG_HK/output/"

st_time = time.time()
# Parameters
U = 4.0  
t = 1.0
mu = 0  # Chemical potential (often set to 0 for simplicity at half filling)
chi_max = 3200  # Maximum bond dimension
sweeps = 40  # Number of DMRG sweeps
k_num = 10

L_list = [10, 16, 22, 28, 34, 40]
GS_list = np.zeros(len(L_list))

def run_dmrg_half_filling(L, U, t, mu, chi_max, sweeps, hc_flag=True):
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
    initial_state = ['up', 'down'] * (L // 2) if L % 2 == 0 else ['up', 'down'] * (L // 2) + ['up']
    psi = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc='finite')

    dmrg_params = {
        'mixer': True,  # Enables mixing for better convergence in difficult phases
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': 1.e-10
        },
        'max_sweeps': sweeps,
    }

    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E0, psi = eng.run()

    return E0

time_list = np.zeros(len(L_list))
for i, L in enumerate(L_list):
    print(L)
    st_time_ind = time.time()
    energy_arr = np.zeros(k_num)
    for ky in range(-int(k_num/2), int(k_num/2)):
        t_star = t*np.exp(1j*ky/k_num*np.pi/L)
        ground_state_energy = run_dmrg_half_filling(L, U, t_star, mu, chi_max, sweeps)
        energy_arr[ky+int(k_num/2)] = ground_state_energy
    GS_list[i] = energy_arr.mean()/L
    time_list[i] = time.time() - st_time_ind
    print(time_list[i])
data_dict = {}
data_dict['GS_list'] = GS_list
data_dict['L_list'] = L_list
data_dict['U'] = U
data_dict['t'] = t
data_dict['mu'] = mu
data_dict['chi_max'] = chi_max
data_dict['sweeps'] = sweeps
data_dict['duration'] = time.time() - st_time
data_dict['k_num'] = k_num
data_dict['time_list'] = time_list
print(data_dict['duration'])
#Give file a unique name with timestamp and unique serial number

filename = '1D_OHK_DMRG_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl'
with open(out_dir+filename, 'wb') as f:
    pickle.dump(data_dict, f)
