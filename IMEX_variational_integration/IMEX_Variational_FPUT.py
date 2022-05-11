#IMEX Variational integrator designed to handle the Fermi-Pasta-Ulam-Tsingou problem,
#where m linear stiff springs and m nonlinear soft springs are aligned in a chain with
#fixed endpoints.
#A discrete integral of the Lagrangian is considered with the implicit midpoint quadrature
#rule for the fast force and an explicit trapezoidal quadrature is used for the slow force.
#This solver is a three step implementation of the method obtained from the discrete
#Euler-Lagrange equations of such discrete Lagrangian.

import numpy as np
import scipy.sparse as sparse

def create_matrices(omega_sq,m,h):
    first_diagonal_Omega=np.zeros(m)
    second_diagonal_Omega=omega_sq*np.ones(m)
    diagonal_Omega=np.concatenate((first_diagonal_Omega,second_diagonal_Omega),axis=0)
    Omega=sparse.dia_matrix((diagonal_Omega,[0]),shape=(2*m,2*m))
    first_diagonal_A=np.ones(m)
    second_diagonal_A=(4/(4+omega_sq*h**2))*np.ones(m)
    diagonal_A=np.concatenate((first_diagonal_A,second_diagonal_A),axis=0)
    A=sparse.dia_matrix((diagonal_A,[0]),shape=(2*m,2*m))
    return Omega,A

def grad_U(q):
    m_2=len(q)
    m=m_2//2
    #Add a dummy element for indexes to not be painfully confusing
    q=np.concatenate(([0],q),axis=0) 
    partials_U=np.zeros(m_2+1)
    for i in range(1,m_2+1): #From 1 to 2m
        for j in range(1,m): #From 1 to m-1
            if j+1==i:
                partials_U[i]+=4*(q[i]-q[m+i]-q[i-1]-q[m+i-1])**3
            elif m+j+1==i:
                partials_U[i]-=4*(q[i-m]-q[i]-q[i-m-1]-q[i-1])**3
            elif j==i:
                partials_U[i]-=4*(q[i+1]-q[m+i+1]-q[i]-q[m+i])**3
            elif m+j==i:
                partials_U[i]-=4*(q[i-m+1]-q[i+1]-q[i-m]-q[i])**3
        if i==1:
            partials_U[i]+=4*(q[1]-q[m+1])**3
        elif i==m+1:
            partials_U[i]-=4*(q[1]-q[m+1])**3
        elif i==m or i==m_2:
            partials_U[i]+=4*(q[m]+q[m_2])**3
    #removing dummy element
    partials_U=partials_U[1:]
    return partials_U
                
        
def IMEX_FPUT_1_step(m,qn,pn,h,omega_sq):
    Omega,A=create_matrices(omega_sq=omega_sq,m=m,h=h)
    pnp=pn-0.5*h*grad_U(qn)
    qn1=A@(qn+h*pnp-0.25*h**2*Omega@qn)
    pn1m=pnp-0.5*h*Omega@(qn1+qn)
    pn1=pn1m-0.5*h*grad_U(qn1)
    return qn1,pn1


    
