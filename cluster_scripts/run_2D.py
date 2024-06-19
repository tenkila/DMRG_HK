#!/usr/bin/env python3

import numpy as np
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.models.lattice import Square
from tenpy.algorithms import dmrg
from tenpy.models.model import CouplingModel, CouplingMPOModel
import pickle, time

out_dir = "/home/tenkila2/scratch/DMRG_out/"

st_time = time.time()
# Parameters
U = 10.0  
t = 1.0
mu = 0  # Chemical potential (often set to 0 for simplicity at half filling)
chi_max = 2000  # Maximum bond dimension
sweeps = 100  # Number of DMRG sweeps
k_num = 10

Lx_list = [2, 4, 6, 8]
Ly_list = [2, 4]
GS_list = np.zeros(shape=(len(Lx_list), len(Ly_list)))



class CustomFermiHubbardModel(CouplingMPOModel):
    def __init__(self, params):
        self.tx = params.pop('tx', 1.0)
        self.ty = params.pop('ty', 1.0)
        self.U = params.pop('U', 1.0)
        self.mu = params.pop('mu', 0.0)
        super().__init__(params)

    def init_terms(self, params):
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            if dx[0] != 0:  # Hopping in the x-direction
                self.add_coupling(self.tx, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
                self.add_coupling(self.tx, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
            if dx[1] != 0:  # Hopping in the y-direction
                self.add_coupling(self.ty, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
                self.add_coupling(self.ty, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(self.U, u, 'NuNd')
            self.add_onsite(self.mu, u, 'Ntot')
            
def run_dmrg_2d(Lx, Ly, U, tx, ty, mu, chi_max, sweeps, charge=0):
    # Define the lattice and SpinHalfFermionSite without any conservation law for simplicity
    site = SpinHalfFermionSite()
    lattice = Square(Lx, Ly, site, bc=['open', 'periodic'])

    # Set up the model parameters
    model_params = {
        'lattice': lattice,
        'tx': tx,  # Hopping amplitude in the x-direction
        'ty': ty,  # Hopping amplitude in the y-direction
        'U': U,
        'mu': mu
    }

    model = CustomFermiHubbardModel(model_params)

    # Initialize with an alternating pattern of filled and empty sites appropriate for 2D
    initial_state = []
    for x in range(Lx):
        for y in range(Ly):
            if (x + y) % 2 == 0:
                initial_state.append('up')
            else:
                initial_state.append('down')

    # Depending on charge modify the last site
    if charge == 1:
        initial_state[-1] = 'full'
    elif charge == -1:
        initial_state[-1] = 'empty'

    psi = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc='finite')

    dmrg_params = {
        'mixer': True,
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': 1.e-5
        },
        'max_sweeps': sweeps,
    }

    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E0, psi = eng.run()
    return E0

for i, Lx in enumerate(Lx_list):
    for j, Ly in enumerate(Ly_list):
        print(Lx, Ly)
        energy_arr = np.zeros(k_num)
        for ky in range(-int(k_num/2), int(k_num/2)):
            ty = t*np.exp(1j*ky/k_num*np.pi)
            tx = t
            ground_state_energy = run_dmrg_2d(Lx, Ly, U, tx, ty, mu, chi_max, sweeps)/Ly
            energy_arr[ky+int(k_num/2)] = ground_state_energy
        GS_list[i,j] = energy_arr.mean()/Lx
        
data_dict = {}
data_dict['GS_list'] = GS_list
data_dict['Lx_list'] = Lx_list
data_dict['Ly_list'] = Ly_list
data_dict['U'] = U
data_dict['t'] = t
data_dict['mu'] = mu
data_dict['chi_max'] = chi_max
data_dict['sweeps'] = sweeps
data_dict['duration'] = time.time() - st_time
data_dict['k_num'] = k_num
print(data_dict['duration'])
#Give file a unique name with timestamp and unique serial number

filename = '2D_OHK_DMRG_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl'
with open(out_dir+filename, 'wb') as f:
    pickle.dump(data_dict, f)

