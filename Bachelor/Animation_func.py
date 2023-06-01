import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML
from numba import jit, types
matplotlib.rcParams['animation.embed_limit'] = 2**128


#===============================================================================
# Def of the animation func:
#===============================================================================


def animate_system(elementary_charge, x_axis, potential, w_anim, r, title = None, figsize_input = [15,7],
                    limits = [-1e-8,8.5e-7,-8000,8000], t_dep_pot=False):

        #Making subplot and setting figsize
    fig, ax = plt.subplots(figsize=(figsize_input[0],figsize_input[1]))
    
    l_1 = ax.plot([], [])[0]
    l_2 = ax.plot([], [], linewidth=3, color="black", label="V")[0]
    

    #Setting labels and title for the system.
    ax.set_xlabel('Length')
    ax.set_ylabel('Height')
    if title == None:
        ax.set_title(f'Animated plot')
    else:
        ax.set_title(title, fontsize= 15)
        
    
    ax.axis([limits[0],limits[1],limits[2],limits[3]])

    
    ax2 = ax.twinx()
    ax2.set_ylabel("$V(x)$")
    #ax2.legend(loc="upper right")
    v_max = np.min(potential) + 1.1 * (np.max(potential) - np.min(potential)) + 1 # + 1 if v = const
    x_ext = np.concatenate(([x_axis[1]], x_axis, [x_axis[-1]]))
    if t_dep_pot == False:

        v_ext = np.concatenate(([v_max/elementary_charge], r*potential, [v_max/elementary_charge]))
        ax2.plot(x_ext, v_ext, linewidth=3, color="black", label="V")
        

        #A function that plots all particles for frame i.
        #This function is run for all frames (see FuncAnimation function)
        def animate_system_frame(i):
            l_1.set_data(x_axis, w_anim[i])

    elif t_dep_pot == True:

        def animate_system_frame(i):
            l_1.set_data(x_axis, w_anim[i])
            v_ext = np.concatenate(([v_max*r], r*potential[i], [v_max*r]))
            l_2.set_data(x_ext, v_ext)

    FPS = np.arange(0,len(w_anim[:,0]),1)
    ani = matplotlib.animation.FuncAnimation(fig, animate_system_frame, frames= FPS)

    plt.close()

    HTML(ani.to_jshtml())
    return ani

