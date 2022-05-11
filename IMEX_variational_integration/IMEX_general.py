#This file contains the implementation of the implicit-explicit variational time integrator presented by Stern and Grinspun, 2009.
#The solver is obtained by the discretization of a Lagragian containing a potential that can be splitted into fast and slow
#components. This is done by applying the midpoint quadrature rule to the fast potential and the trapezoidal quadrature rule to
#the slow potential, leading to an implicit-explicit solver (for fast and slow forces respectively).

import numpy as np

#We consider a Lagrangian of the form: L(q,q')=0.5*q'T M q' -V(q), where M is a mass matrix and V a potential

#Trapezoidal discrete lagrangian
#Discrete Euler-Lagrange equations lead to explicit Störmer/Verlet method
def L_mid(q0,q1,h,M,V):
    return (h/2)*np.transpose((q1-q0)/h)*M*((q1-q0/h))-h*(0.5*(V(q0)+V(q1)))

#Midpoint discrete lagrangian
#Discrete Euler-Lagrange equations lead to implicit midpoint method
def L_mid(q0,q1,h,M,V):
    return (h/2)*np.transpose((q1-q0)/h)*M*((q1-q0/h))-h*V(0.5*(q0+q1))

#If we have the particular case of V(q)=U(q)+W(q), where U is a slow potential and W is a fast potential, we define

#IMEX discrete Lagrangian
def L_IMEX(q0,q1,h,M,U,W):
    return (h/2)*np.transpose((q1-q0)/h)*M*((q1-q0/h)-h*(0.5*(U(q0)+U(q1)))
)-h*W(0.5*(q0+q1))


def IMEX_integrator(M_inv, U_grad, W_grad, q0,p0):

    
