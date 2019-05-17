#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: katerina

Reference:
Wolf, A., Swift, J. B., Swinney, H. L., & Vastano, J. A. (1985).
Determining Lyapunov exponents from a time series. 
Physica D: Nonlinear Phenomena, 16(3), 285-317. 

In this code the 3 Lyapunox Exponents of Lorenz63 system are calculated, 
Based on Wolf's paper algoritm.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D


def Lor(X,t):
    print
    x, y, z = X
    return np.array([s * (y - x),
            r * x - y - x * z,
            x * y - b * z])

    
def JacLor(t,X):
    x, y, z = X
    Jac = np.array([[-s, s, 0],
                  [r - z, -1, -x],
                  [y, x, -b]])
    return Jac


if __name__ == "__main__" :

    # Model's Parameters
    s, b, r = (16, 4, 45.92)
    print("System = Lorenz63\n", "s, b, r = ", s, ",", b, ",", r)
    
    # ADAPTATION SIMULATION
    #initial value
    x=1; y=0; z=0

    X=np.array([x,y,z])
    adaptTime=200
    adaptInt=0.01  #This controls only for the returned values, not the calculation
    Tadapt=np.arange(0,adaptTime,adaptInt)
    Yadapt=integrate.odeint(Lor,X,Tadapt)
    
    # SIMULATION FOR LEs CALCULATION
    X0=Yadapt[-1]  #initial conditions after some adaptation
    print("X0", X0)

    # Simulation for the calculation    
    tBegin = 0
    tEnd = 88
    deltat = 0.0001  # time interval between Jacobian calculations
    print("tEnd = ", tEnd, ", deltat = ", deltat )
    T = np.arange(tBegin, tEnd, deltat)

##     ##
    Q = np.identity(len(X0)) * 0.1
    print(Q)
    X = X0
    
    LCE_T = []
    norm_T = []
    for t in T:
        X = integrate.odeint( Lor, X, (0, deltat) )[-1]      
        B = Q + deltat * np.dot( JacLor(t, X), Q )
        Q, R = np.linalg.qr(B) # Factor the matrix B as qr, where q is orthonormal and r is upper-triangular.
        LCE_T.append( np.log2( np.abs(np.diag(R)) ) / deltat )
    LCE_T = np.array(LCE_T)
 ##    ##

    # Discard initial timestesp
    i_init = 10  
    LCEv = np.cumsum(LCE_T[i_init:, : ], 0) / (T / deltat + deltat)[i_init:, None]
    print("\nSystems's Lyapunov Exponents = ", LCEv[-1])
    
    # Plot    
    plt.plot(T[i_init : ], LCEv)
    plt.xlabel("Time")
    plt.show()
    