#Performance comparison of IMEX Variational and Stormer-Verlet integrators
#with a Fermi-Pasta-Ulam-Tsingou problem

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from IMEX_Variational_and_Stormer_Verlet_FPUT import IMEX_FPUT_1_step, Hamiltonian_FPUT, I_oscillatory_energy
from IMEX_Variational_and_Stormer_Verlet_FPUT import Stormer_Verlet_FPUT_1_step, T_kinetic_energies


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

def static_subplots_energy(list_simulations_IMEX,list_simulations_SV,list_h_IMEX,list_h_SV):
    #len(list_simulations_SV_h)=3
    #len(list_simulations_IMEX_h)=6
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

    #Plotting
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
    plt.savefig("Comparison_energies.png", bbox_inches="tight")
    plt.show()


def run_simulation(T_end=200,h=0.03, method="IMEX"):
    #Setting problem parameters, assuming initial time 0
    m=3
    omega=50
    omega_sq=omega**2

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

def get_comparison_energies():
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

    static_subplots_energy(list_simulations_IMEX,list_simulations_SV,list_h_IMEX,list_h_SV)

def get_comparison_error():
    T_end=20
    ground_truth_IMEX=run_simulation(T_end=T_end,h=0.0001, method="IMEX")
    exact_q_IMEX,exact_p_IMEX=ground_truth_IMEX[-1].q,ground_truth_IMEX[-1].p


    ground_truth_SV=run_simulation(T_end=T_end,h=0.0001, method="SV")
    exact_q_SV,exact_p_SV=ground_truth_SV[-1].q,ground_truth_SV[-1].p

    errors_q_IMEX=[]; errors_p_IMEX=[]
    errors_q_SV=[];  errors_p_SV=[]

    min_h=0.01
    max_h=0.1
    step_h=0.005
    hs=np.arange(min_h,max_h+step_h,step_h)
    
    for h in hs:
        simulation_IMEX=run_simulation(T_end=T_end, h=h, method="IMEX") 
        q_IMEX,p_IMEX=simulation_IMEX[-1].q,simulation_IMEX[-1].p  
        errors_q_IMEX.append(np.linalg.norm(x=q_IMEX-exact_q_IMEX,ord=2))
        errors_p_IMEX.append(np.linalg.norm(x=p_IMEX-exact_p_IMEX,ord=2))

        simulation_SV=run_simulation(T_end=T_end, h=h, method="SV") 
        q_SV,p_SV=simulation_SV[-1].q,simulation_SV[-1].p  
        errors_q_SV.append(np.linalg.norm(x=q_SV-exact_q_SV,ord=2))
        errors_p_SV.append(np.linalg.norm(x=p_SV-exact_p_SV,ord=2))
    
    fig, axs = plt.subplots(2,1, sharex=True)
    axs[0].plot(hs,errors_q_IMEX,label="Error IMEX")
    axs[0].plot(hs,errors_q_SV,label="Error SV")
    axs[0].set_title(f"L2 error of q")

    axs[1].plot(hs,errors_p_IMEX,label="Error IMEX")
    axs[1].plot(hs,errors_p_SV,label="Error SV")
    axs[1].set_title(f"L2 error of p")

    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")
    fig.tight_layout()
    plt.savefig("Comparison_errors.png", bbox_inches="tight")

if __name__=="__main__":
   get_comparison_energies()
   get_comparison_error()