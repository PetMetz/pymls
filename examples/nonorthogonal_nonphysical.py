# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 16:27:19 2022

@author: UT
"""

# 3rd party
import numpy as np

# package
from pymls import Lattice, Dislocation, Stroh, MLS
from pymls.elastic import cij_from_group
from pymls.toolbox import abt


# - 1. crystal lattice
scalar = (1, 1, 1, 60, 90, 120)

# - 2. slip system
hkl = np.array((0,0,1))  # BCC slip plane
uvw = np.array((1,0,0))  # burgers vector
l   = np.array((0,1,0))  # dislocation line vector
phi = abt(uvw, l, degrees=True) # 90 degrees == edge dislocation

# - 3. elastic constituents
C = cij_from_group(116.3, 64.8, 30.9, group='m-3m') # GPa

# - 4. class instances
L = Lattice.from_scalar( scalar ) 
D = Dislocation(lattice=L, hkl=hkl, uvw=uvw, phi=phi, SGno=None)
S = Stroh(C) # captures characteristic elastic matrix and eigensolution
I = MLS(dislocation=D, cij=C) # captures sum computation

# - 5. viz
fig, ax = D.visualize()




# - 
def MLS_M(lattice):
    """ eqn. 1 """
    D = lattice
    R = lattice.reciprocal
    cos = np.cos(D.angles * np.pi/180)
    sin = np.sin(D.angles * np.pi/180)
    cosstar = np.cos(R.angles * np.pi/180)
    M = np.array((
        (1/D.a                 , 0                 , 0  ),
        (-cos[2] / (D.a*sin[2]), 1 / (D.b * sin[2]), 0  ),
        (R.a * cosstar[1]      , R.b * cosstar[0]  , R.c)
        ))
    return M
    


def vol_from_scalar(a,b,c,al,be,ga):
    cosa = np.cos(al*np.pi/180)
    cosb = np.cos(be*np.pi/180)
    cosg = np.cos(ga*np.pi/180)
    return a*b*c*np.sqrt(1 + 2*cosa*cosb*cosg - cosa**2 - cosb**2 - cosg**2)


# =============================================================================
# 
# M = MLS_M(L)
# L = Lattice(M)
# D = Dislocation(lattice=L, hkl=hkl, uvw=uvw, phi=phi)
# C2 = MLS(D, C)
# 
# =============================================================================
