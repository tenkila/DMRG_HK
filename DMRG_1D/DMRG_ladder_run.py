#!/usr/bin/env python3

import numpy as np
from tenpy.networks.mps import MPS
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.algorithms import dmrg
import pickle, time

out_dir = "/home/tenkila2/scratch/DMRG_out/"

st_time = time.time()
# Parameters
L = 8  # Number of lattice sites (ensure it is even for a simple setup)
U = 0.0  # On-site interaction strength
U_start = 3.0
t = 1
mu = 0  # Chemical potential (often set to 0 for simplicity at half filling)
chi_max = 200  # Maximum bond dimension
sweeps = 100  # Number of DMRG sweeps


n_rungs_list = range(1, 2, 1)
L_list = range(10, 20, 20)
GS_list = np.zeros(shape=(len(n_rungs_list), len(L_list)))



def low_U_ladder(L, U, U_start, n_rungs, t, mu, chi_max, sweeps, hc_flag=False):
    
    model_params = {
            'L': L,
            'U': U_start,
            't':t,
            'mu': mu,
            'bc_MPS': 'finite',
            'bc_x':'periodic',
            'explicit_plus_hc':hc_flag
        }

    model = FermiHubbardModel(model_params)
    
    initial_state = ['up', 'down'] * (L // 2) if L % 2 == 0 else ['up', 'down'] * (L // 2) + ['up']
    psi = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc='finite')
    
    for U_new in np.linspace(U_start, U, n_rungs):
        model_params = {
            'L': L,
            'U': U_new,
            't':t,
            'mu': mu,
            'bc_MPS': 'finite',
            'bc_x':'periodic',
            'explicit_plus_hc':hc_flag
        }

        model = FermiHubbardModel(model_params)


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

for j, L in enumerate(L_list):
    for i, n_rungs in enumerate(n_rungs_list):
        GS_list[i, j] = low_U_ladder(L, U, U_start, n_rungs, t, mu, chi_max, sweeps)/L
        print(L, n_rungs)


data_dict = {}
data_dict['GS_list'] = GS_list
data_dict['L_list'] = L_list
data_dict['n_rungs_list'] = n_rungs_list
data_dict['U'] = U
data_dict['U_start'] = U_start
data_dict['t'] = t
data_dict['mu'] = mu
data_dict['chi_max'] = chi_max
data_dict['sweeps'] = sweeps
data_dict['duration'] = time.time() - st_time
print(data_dict['duration'])
#Give file a unique name with timestamp and unique serial number

filename = 'U_ladder_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl'
with open(out_dir+filename, 'wb') as f:
    pickle.dump(data_dict, f)


