import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML
from numba import jit, types
matplotlib.rcParams['animation.embed_limit'] = 2**128
from Wave_func_propagation import *

@jit(forceobj=True, fastmath=True, error_model='numpy', parallel=True)
def  probability_density(wave, Desired_t, dx, pot_size, x_axis):

    left_barier, righ_barier = int(len(x_axis)/2 - pot_size[0]/2), int(len(x_axis)/2 + pot_size[0]/2)


    probability_tot = np.abs(wave[Desired_t])**2
    probability_trans = np.abs(wave[Desired_t,righ_barier:])**2
    probability_reflec = np.abs(wave[Desired_t,:left_barier])**2

    normalisation_check1 = np.sum(probability_tot)*dx
    normalisation_check2 = np.sum(probability_trans)*dx + np.sum(probability_reflec)*dx
    normalisation_trans = np.sum(probability_trans)*dx
    normalisation_reflec = np.sum(probability_reflec)*dx   

    

    return probability_tot, normalisation_trans, normalisation_reflec

# @jit(forceobj=True, fastmath=True, error_model='numpy', parallel=True)
def transmition(pot_barrier, Energis, pot_size, sigma_x, x_s, L, hbar, m, K, C, t_dep_pot=False):

    k_0 = np.sqrt(2*m*Energis[:])/hbar
    
    transmition_probabilitis = np.zeros(len(Energis))
    #print(transmition_probabilitis.shape)
    
    for i in range(len(Energis)):
        
        k = k_0[i]
        print('Round:',i+1)
        Psi_t, psi_imag_t, psi_real_t, x_axis, dx, T, N_t, harm_pot_animat  = Psi_propagation(Psi_initial, pot_barrier,  sigma_x, x_s, hbar,
                                                                              m, L, k, K, C, pot_size, t_dep_pot)
        #print(Psi_t.shape)
        probability_tot, transmition_probabilitis[i], normalisation_reflec = probability_density(Psi_t, -1, dx, pot_size, x_axis)
        

    return transmition_probabilitis

@jit(forceobj=True, fastmath=True, error_model='numpy', parallel=True)
def transmition_for_omega(pot_barrier, freq, sigma_x, x_s, hbar, m, L, k, K, C, barrier_size, well_size, barrier_hight, E_resonance, phase):

    transmition_probabilitis = np.zeros(len(freq))

    for i in range(len(freq)):
        print('Round:',i+1)
        pot_size = np.array([barrier_size, well_size, barrier_hight, E_resonance, freq[i]], phase)

        Psi_t, psi_imag_t, psi_real_t, x_axis, dx, T, N_t, harm_pot_animat  = Psi_propagation(Psi_initial, pot_barrier,  sigma_x, x_s, hbar,
                                                                              m, L, k, K, C, pot_size, t_dep_pot=True)

        probability_tot, transmition_probabilitis[i], normalisation_reflec = probability_density(Psi_t, -1, dx, pot_size, x_axis)
        

    return transmition_probabilitis


@jit(forceobj=True, fastmath=True, error_model='numpy', parallel=True)
def transmition_for_V1(pot_barrier, V1, sigma_x, x_s, hbar, m, L, k, K, C, barrier_size, well_size, barrier_hight, omega, phase):

    transmition_probabilitis = np.zeros(len(V1))

    for i in range(len(V1)):
        # print('Round:',i+1)
        pot_size = np.array([barrier_size, well_size, barrier_hight, V1[i], omega, phase])

        Psi_t, psi_imag_t, psi_real_t, x_axis, dx, T, N_t, harm_pot_animat  = Psi_propagation(Psi_initial, pot_barrier,  sigma_x, x_s, hbar,
                                                                              m, L, k, K, C, pot_size, t_dep_pot=True)

        probability_tot, transmition_probabilitis[i], normalisation_reflec = probability_density(Psi_t, -1, dx, pot_size, x_axis)
        

    return transmition_probabilitis


def transmition_for_phase(pot_barrier, phase, sigma_x, x_s, hbar, m, L, k, K, C, barrier_size, well_size, barrier_hight, omega, V1):

    transmition_probabilitis = np.zeros(len(phase))

    for i in range(len(phase)):
        print('Round:',i+1)
        pot_size = np.array([barrier_size, well_size, barrier_hight, V1, omega, phase[i]])

        Psi_t, psi_imag_t, psi_real_t, x_axis, dx, T, N_t, harm_pot_animat  = Psi_propagation(Psi_initial, pot_barrier,  sigma_x, x_s, hbar,
                                                                              m, L, k, K, C, pot_size, t_dep_pot=True)

        probability_tot, transmition_probabilitis[i], normalisation_reflec = probability_density(Psi_t, -1, dx, pot_size, x_axis)
        

    return transmition_probabilitis