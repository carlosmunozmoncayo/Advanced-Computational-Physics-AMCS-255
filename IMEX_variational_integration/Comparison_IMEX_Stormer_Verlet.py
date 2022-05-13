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
    H+=0.5*omega_sq*q[m]**2+0.25*((q[1]-q[m+1])**4+(q[m]+q[m_2])**4)
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
        state.I, state.I_list= I_oscillatory_energy(q=q,p=p,omega_sq=omega_sq)

def run_simulation():
    #Setting problem parameters, assuming initial time 0
    m=3
    omega=50
    omega_sq=omega**2
    T_end=1
    h=0.001 #Time step
    
    #Initializing states
    q0=np.array([1,0,0,1/omega,0,0])
    qn_IMEX=np.copy(q0)
    qn_SV=np.copy(q0)
    
    p0=np.array([1,0,0,1,0,0])
    pn_IMEX=np.copy(p0)
    pn_SV=np.copy(p0)
    
    #Vector of times
    times=np.arange(0,T_end+h,h)
    states_IMEX=[State(q0,p0,0.0,omega_sq)]
    states_SV=[State(q0,p0,0.0,omega_sq)]

    for i in range(1,len(times)):
        qn_IMEX, pn_IMEX=IMEX_FPUT_1_step(qn=qn_IMEX,pn=pn_IMEX,h=h,omega_sq=omega_sq)
        states_IMEX.append(State(qn_IMEX,pn_IMEX,times[i],omega_sq))
        qn_SV, pn_SV=Stormer_Verlet_FPUT_1_step(qn=qn_SV,vn=pn_SV,h=h,omega_sq=omega_sq)
        states_SV.append(State(qn_SV,pn_SV,times[i],omega_sq))
   # print(qn_IMEX)
   # print(qn_SV)
    for i in range(0,len(states_IMEX)):
        print(states_IMEX[i].I)
    #print(states_IMEX[-1].H)
if __name__=="__main__":
    run_simulation()
