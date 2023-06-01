import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML
from numba import jit, types
from scipy.sparse import diags
from scipy.integrate import quad
matplotlib.rcParams['animation.embed_limit'] = 2**128

#===============================================================================



def Fourier_transformed(Wave, Desired_t, Dx, hbar, mass, x_axis, pot_size):

    left_barier, right_barier = int(len(x_axis)/2  - pot_size[0]/Dx - pot_size[1]/(Dx*2)), int(len(x_axis)/2  + pot_size[0]/Dx + pot_size[1]/(Dx*2))
    x_reflected = x_axis[:left_barier]
    x_transmited = x_axis[right_barier:]
    # print(int(len(x_axis)/2 - 500), left_barier*dx, right_barier*dx)

    Transformed_total = np.fft.fft(Wave[Desired_t,:],n=1000000)
    Transformed_reflected = np.fft.fft(Wave[Desired_t,:left_barier],n=1000000)
    Transformed_transmited = np.fft.fft(Wave[Desired_t,right_barier:],n=1000000)

    N = len(Transformed_total)
    freq = np.fft.fftfreq(N, Dx)

    firstNegInd = np.argmax(freq < 0)
    freq = freq[:firstNegInd]

    Transformed_reflected = Transformed_reflected[::-1]
    Transformed_reflected = 2 * Transformed_reflected[:firstNegInd]
    Transformed_transmited = 2 * Transformed_transmited[:firstNegInd]

    
    
    freq = freq*2*np.pi


    
    return Transformed_reflected, Transformed_transmited, Transformed_total, freq, #freq_r#, Excpeted_Energi_tot