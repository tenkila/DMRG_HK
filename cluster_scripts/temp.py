#!/usr/bin/env python3

import numpy as np
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.models.lattice import Chain
from tenpy.models import lattice
from tenpy.algorithms import dmrg
from tenpy.models.model import CouplingModel, CouplingMPOModel

import pickle, time
out_dir = "/home/tenkila2/scratch/DMRG_out/"

st_time = time.time()
# Parameters
t = 1
k_num = 10
chi_max = 200  # Maximum bond dimension
sweeps = 100  # Number of DMRG sweeps
U_list = np.array([0.1, 0.2, 0.5, 1, 2, 4, 6, 8])
norb_list = np.array([6,8, 10, 12, 16, 20])
bc_tag = 'open'

class CustomFermiHubbardModel(CouplingMPOModel):
    
    def __init__(self, params):
        #Extract the additional parameters from the input data
        self.t = params.pop('t', 1.0)
        self.U = params.pop('U', 1.0)
        self.mu = params.pop('mu', 0.0)
        super().__init__(params)

    def init_terms(self, params):
        #Add in custom terms to your hamiltonian
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            if dx[0] != 0:  # Hopping in the x-direction
                self.add_coupling(self.t, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
                self.add_coupling(self.t, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
                
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(self.U, u, 'NuNd')
            self.add_onsite(-self.mu, u, 'Ntot')

def run_dmrg_half_filling(L, U, t, mu, chi_max, sweeps, hc_flag=False, charge=0):
    
    site = SpinHalfFermionSite() 
    lattice = Chain(L, site, bc=bc_tag)
    
    model_params = {
        'lattice': lattice,
        'U': U,
        't':t,
        'mu': mu
    }

    model = CustomFermiHubbardModel(model_params)

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

    return E0 #, psi.expectation_value('Ntot')


time_list = np.zeros(shape=(len(U_list), len(norb_list)))
raw_energy_dat = np.zeros(shape=(len(U_list), len(norb_list), 2, k_num))

for i, U in enumerate(U_list):
    for j, norb in enumerate(norb_list):
        mu = U/2
        st_time = time.time()
        energy_arr = np.zeros(k_num)
        for kx in range(-int(k_num/2), int(k_num/2)):
            t_star = np.exp(1j*kx/k_num*np.pi)
            raw_energy_dat[i,j,0,kx+int(k_num/2)] = run_dmrg_half_filling(norb, U, t_star, mu, chi_max, sweeps,charge=0)
            raw_energy_dat[i,j,1,kx+int(k_num/2)] = run_dmrg_half_filling(norb, U, t_star, mu, chi_max, sweeps,charge=1)

        time_list[i, j] = time.time() - st_time
        print(U, norb, time_list[i,j])

data_dict = {}
data_dict['U_list'] = U_list
data_dict['norb_list'] = norb_list
data_dict['raw_energy'] = raw_energy_dat
data_dict['t'] = t
data_dict['chi_max'] = chi_max
data_dict['sweeps'] = sweeps
data_dict['duration'] = time.time() - st_time


filename = 'Charge_gap_1D_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl'
with open(out_dir+filename, 'wb') as f:
    pickle.dump(data_dict, f)
