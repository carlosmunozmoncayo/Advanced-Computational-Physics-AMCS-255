#Performance comparison of IMEX Variational and Stormer-Verlet integrators
#with a Fermi-Pasta-Ulam-Tsingou problem

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from IMEX_Variational_and_Stormer_Verlet_FPUT import IMEX_FPUT_1_step
from IMEX_Variational_and_Stormer_Verlet_FPUT import Stormer_Verlet_FPUT_1_step

#####
#Defining energy estimates for each state
####
def Hamiltonian_FPUT(q,p,omega_sq):
    H=0
    m_2=len(q)
    m=m_2//2
    
    #Add a dummy element for nicer indexes 
    q=np.concatenate(([0],q),axis=0)
    p=np.concatenate(([0],p),axis=0)
    
    #Computing Hamiltonian
    H+=0.5*np.sum(p**2)
    for i in range(1,m): #From 1 to m-1
        H+=0.5*omega_sq*q[m+i]**2
        H+=0.25*(q[i+1]-q[m+i+1]-q[i]-q[m+i])**4
    H+=0.5*omega_sq*q[m_2]**2+0.25*((q[1]-q[m+1])**4+(q[m]+q[m_2])**4)
    return H

def I_oscillatory_energy(q,p,omega_sq):
    m_2=len(q)
    m=m_2//2
    I_list=0.5*(p[m:]**2+omega_sq*q[m:]**2)
    I=np.sum(I_list)
    return I,I_list

def T_kinetic_energies(p):
    m=len(p)//2
    #Kinetic energy of mass centre motion
    T1=0.5*np.sum(p[:m]**2)
    #Kinetic energy of relative motion of masses joined by stiff springs
    T2=0.5*np.sum(p[m:]**2)
    return T1,T2

#####
#A grid function norm to measure error
#####
def q_norm_grid_function(x,h,q=2):
    if q=="max":
        return np.sum(np.abs(x))
    else:
        return (h*np.sum(np.abs(x)**q))**(1/q)

#####
#A class to store our states
####
class State:
    def __init__(state, q, p, time, omega_sq):
        state.q=q
        state.p=p
        state.time=time
        state.H=Hamiltonian_FPUT(q=q,p=p,omega_sq=omega_sq)
        state.T1,state.T2=T_kinetic_energies(p)
        state.I, [state.I1, state.I2, state.I3]= I_oscillatory_energy(q=q,p=p,omega_sq=omega_sq)

#####
#Plotting results
#####

def static_plots(states_IMEX,states_SV):
    #Extracting data from objects (2 loops for clarity)
    times=[]; H_IMEX=[]; I_IMEX=[]
    I1_IMEX=[]; I2_IMEX=[]; I3_IMEX=[]
    for state in states_IMEX:
        times.append(state.time)
        H_IMEX.append(state.H)
        I_IMEX.append(state.I)
        I1_IMEX.append(state.I1)
        I2_IMEX.append(state.I2)
        I3_IMEX.append(state.I3)
    H_SV=[]; I_SV=[]
    I1_SV=[]; I2_SV=[]; I3_SV=[]
    for state in states_SV:
        H_SV.append(state.H)
        I_SV.append(state.I)
        I1_SV.append(state.I1)
        I2_SV.append(state.I2)
        I3_SV.append(state.I3)

    # plot lines
    plt.plot(times, H_IMEX,label = "H IMEX")
    plt.plot(times, I_IMEX, label = "I IMEX")
    plt.plot(times, I1_IMEX, label = "I1 IMEX")
    plt.plot(times, I2_IMEX, label = "I2 IMEX")
    plt.plot(times, I3_IMEX, label = "I3 IMEX")  
    plt.legend()
    plt.show()
    
    plt.plot(times, H_SV,label = "H SV")
    plt.plot(times, I_SV, label = "I SV")
    plt.plot(times, I1_SV, label = "I1 SV")
    plt.plot(times, I2_SV, label = "I2 SV")
    plt.plot(times, I3_SV, label = "I3 SV")  
    plt.legend()
    plt.show()

def static_subplots(list_simulations_IMEX,list_simulations_SV,list_h_IMEX,list_h_SV):
    #len(list_states_SV_h)=3
    #len(list_states_IMEX_h)=6
    #list_simulations_IMEX=[simulation,simulation,...,simulation]
    #simulation=[state at time 0,state at time h,state at time 2h,...]
    
    #Extracting data from objects 
    attributes_allplots_IMEX=[]
    for simulation_IMEX in list_simulations_IMEX:
        times=[]; H_IMEX=[]; I_IMEX=[]
        I1_IMEX=[]; I2_IMEX=[]; I3_IMEX=[]
        for state in simulation_IMEX:
            times.append(state.time)
            H_IMEX.append(state.H)
            I_IMEX.append(state.I)
            I1_IMEX.append(state.I1)
            I2_IMEX.append(state.I2)
            I3_IMEX.append(state.I3)
        attributes_1plot_IMEX=[H_IMEX,I_IMEX,I1_IMEX,I2_IMEX,I3_IMEX,times]
        attributes_allplots_IMEX.append(attributes_1plot_IMEX)

    attributes_allplots_SV=[]
    for simulation_SV in list_simulations_SV:
        times=[]; H_SV=[]; I_SV=[]
        I1_SV=[]; I2_SV=[]; I3_SV=[]
        for state in simulation_SV:
            times.append(state.time)
            H_SV.append(state.H)
            I_SV.append(state.I)
            I1_SV.append(state.I1)
            I2_SV.append(state.I2)
            I3_SV.append(state.I3)
        attributes_1plot_SV=[H_SV,I_SV,I1_SV,I2_SV,I3_SV,times]
        attributes_allplots_SV.append(attributes_1plot_SV)


    text=["H","I","I1","I2","I3"]
    fig, axs = plt.subplots(3, 3)
    #First and second rows
    for i in range(3):
        for j in range(5):
            axs[0,i].plot(attributes_allplots_SV[i][5],attributes_allplots_SV[i][j],label=text[j])
            axs[1,i].plot(attributes_allplots_IMEX[i][5],attributes_allplots_IMEX[i][j],label=text[j])
        axs[0,i].set_title(f"Störmer Verlet h={list_h_SV[i]}")
        axs[1,i].set_title(f"IMEX h={list_h_IMEX[i]}")
    
    for i in range(3,6):
        for j in range(5):
            axs[2,i-3].plot(attributes_allplots_IMEX[i][5],attributes_allplots_IMEX[i][j],label=text[j])
        axs[2,i-3].set_title(f"IMEX h={list_h_IMEX[i]}")

    axs[0,0].legend(loc="upper right")
    fig.tight_layout()
    plt.show()


def run_simulation(h=0.03, method="IMEX"):
    #Setting problem parameters, assuming initial time 0
    m=3
    omega=50
    omega_sq=omega**2
    T_end=200

    if method=="IMEX":
        solver=IMEX_FPUT_1_step
    elif method=="SV":
        solver=Stormer_Verlet_FPUT_1_step

    
    #Initializing states
    qn=np.array([1,0,0,1/omega,0,0])
    pn=np.array([1,0,0,1,0,0])
    
    #Vector of times
    times=np.arange(0,T_end+h,h)
    simulation=[State(qn,pn,0.0,omega_sq)]

    for i in range(1,len(times)):
        qn, pn=solver(qn=qn,pn=pn,h=h,omega_sq=omega_sq)
        simulation.append(State(qn,pn,times[i],omega_sq))
    
    return simulation

if __name__=="__main__":
    list_simulations_IMEX=[]
    list_simulations_SV=[]
    list_h_SV=[0.001,0.01, 0.03]
    #list_h_IMEX=[0.03, 0.1, 0.15, 0.2, 0.25, 0.3]
    list_h_IMEX=[0.001,0.01, 0.03,0.1, 0.15, 0.2]

    for h in list_h_SV:
        simulation_SV=run_simulation(h=h,method="SV")
        list_simulations_SV.append(simulation_SV)
    for h in list_h_IMEX:
        simulation_IMEX=run_simulation(h=h, method="IMEX")
        list_simulations_IMEX.append(simulation_IMEX)

    static_subplots(list_simulations_IMEX,list_simulations_SV,list_h_IMEX,list_h_SV)
