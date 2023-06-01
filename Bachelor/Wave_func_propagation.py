import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML
from numba import jit, types
from scipy.sparse import diags
matplotlib.rcParams['animation.embed_limit'] = 2**128



#===============================================================================
# Initial wave:
#===============================================================================

@jit(forceobj=True, fastmath=True, error_model='numpy', parallel=True)
def Psi_initial(x_axis, sigma_x, x_s, k_0):
    
    '''
    Calculating the initial valuse of the real and imaginere parts of the wave pack and adding it together to get Psi 
    '''
    
    normalisation = 1/(2*np.pi * sigma_x**2)**(1/4)
    gaussian = np.exp(-(x_axis-x_s)**2/(2*sigma_x**2))

    #psi_imag_initial = normalisation * gaussian * np.sin(k_0*x -omega *Dt/2)
    #psi_real_initial = normalisation * gaussian * np.cos(k_0*x)
    
    Psi = normalisation * gaussian * np.exp(1j*(k_0*x_axis))


    psi_imag_initial = np.imag(Psi)
    psi_real_initial = np.real(Psi)

    psi_imag_initial[0] = 0.0
    psi_real_initial[0] = 0.0
    psi_imag_initial[-1] = 0.0
    psi_real_initial[-1] = 0.0



    return Psi, psi_imag_initial, psi_real_initial

# @jit(forceobj=True, fastmath=True, error_model='numpy', parallel=True)
def Harmonic_Potential(x_axis, Potential, potential, dx, dt, t, hbar, matrix, Derivat):


    middel = len(x_axis)/2 *dx
    well = Potential[1]

    const = Potential[4] *dt 

    left_well_edge = middel - well/2
    righ_well_edge = middel + well/2

    left_well_edge_i = np.searchsorted(x_axis, left_well_edge, side='righ')
    righ_well_edge_i = np.searchsorted(x_axis, righ_well_edge, side='righ')

    
    potential[left_well_edge_i : righ_well_edge_i] = Potential[3]*np.cos(const * t + Potential[5])
    Pot = diags(potential*(dt / hbar)).tocsr()

    # for i in range(int(well)):

    #     potential[middel - i] = Potential[3]*np.cos(const * t)
    #     potential[middel + i+1] = Potential[3]*np.cos(const * t)

    #     Pot = diags(potential*(dt / hbar)).tocsr()
    matrix = Pot - Derivat

    return matrix, potential


#===============================================================================
# Wave propagation func:
#===============================================================================

# @jit(forceobj=True, fastmath=True, error_model='numpy', parallel=True)
def Psi_propagation(intial_state, Potential, sigma_x, x_s, hbar, mass, L, k_0, K, C, Potential_size=[0,0,0], t_dep_pot=False):

    N_x = K*int(2*k_0*L) + 1 
    dx = L/(N_x-1)
    v_g = hbar * k_0/mass

     

    x_axis = np.linspace(0.0, L, N_x, dtype=np.float64)

    dt = C * hbar / ((hbar**2/(2*mass*dx**2))+Potential_size[2])
    T = 1.0*L/(2*v_g) # for N_x=15, dt= 0.005... use 1.1
    N_t = int(T/dt)
    time =  np.linspace(0,T,N_t)


    Psi_init, psi_imag_iter, psi_real_iter= intial_state(x_axis, sigma_x, x_s, k_0)
    potential, v_max= Potential(x_axis, dx, time, Potential_size)
    

    #time = np.arange(0,T,Dt)
    harm_pot_animat = np.zeros((int(N_t/20000+3), len(x_axis)), dtype=np.float64, order='C')
    psi_imag = np.zeros((int(N_t/20000+3), len(x_axis)), dtype=np.float64, order='C')
    psi_real = np.zeros((int(N_t/20000+3), len(x_axis)), dtype=np.float64, order='C')
    Psi = np.zeros((int(N_t/20000+3), len(x_axis)), dtype=np.complex128, order='C')
    psi_imag[0,:] =  psi_imag_iter
    psi_real[0,:] =  psi_real_iter
    Psi[0,:] =  Psi_init
    
    

    # defing a matrix to calulat the next time step
    semi_diag = np.full(len(x_axis)-1, 1)*(hbar * dt / (2*mass *dx**2))
    Derivat_diags = [np.full(len(x_axis), -2)*(hbar * dt / (2*mass *dx**2)), semi_diag, semi_diag]

    Pot = diags(potential*(dt / hbar))
    Derivat = diags(Derivat_diags, [0, -1, 1]).tocsr()
    matrix = - Derivat + Pot

    t_counter = 0

    
    if t_dep_pot == False:

        for t in range(N_t-1):

            psi_imag_iter += -matrix.dot( psi_real_iter)
            psi_real_iter += matrix.dot( psi_imag_iter)

            if t%20000 == 0:
                
                psi_imag[t_counter,:] = psi_imag_iter
                psi_real[t_counter,:] = psi_real_iter
                Psi[t_counter,:] = psi_real_iter + 1j * psi_imag_iter 
                t_counter += 1
                
        
    elif t_dep_pot == True:

        for t in range(N_t-1):
            
            
            
            matrix, harm_pot = Harmonic_Potential(x_axis, Potential_size, potential, dx, dt, t, hbar, matrix, Derivat)

            # bruk spars matrix 

            # Pot = diags(harm_pot*(dt / hbar))
            # Derivat = diags(Derivat_diags, [0, -1, 1])
            # matrix = - Derivat + Pot
            

            psi_imag_iter += -matrix.dot(psi_real_iter)
            psi_real_iter += matrix.dot(psi_imag_iter)

            if t%20000 == 0:
                
                psi_imag[t_counter,:] = psi_imag_iter
                psi_real[t_counter,:] = psi_real_iter
                Psi[t_counter,:] = psi_real_iter + 1j * psi_imag_iter 

                harm_pot_animat[t_counter,:] = harm_pot

                t_counter += 1

        harm_pot_animat[-1,:] = harm_pot
    
    psi_imag[-1,:] = psi_imag_iter
    psi_real[-1,:] = psi_real_iter
    Psi[-1,:] = psi_real_iter + 1j * psi_imag_iter 
    

    return Psi, psi_imag, psi_real, x_axis, dx, T, N_t, harm_pot_animat


