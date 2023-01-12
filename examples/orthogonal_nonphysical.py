# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 16:27:19 2022

@author: UT
"""

# 3rd party
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt

# package
from pymls import Lattice, Dislocation, Stroh, MLS
from pymls.elastic import cij_from_group
from pymls.toolbox import abt


plt.close('all')


# - 1. crystal lattice
M = 2 * np.eye(3)

# - 2. slip system
hkl = np.array((0,0,1))  # slip plane normal
uvw = np.array((1,0,0))  # burgers vector
l   = np.array((1,-1,0))  # dislocation line vector
phi = abt(uvw, l, degrees=True) # 90 degrees == edge dislocation

# - 3. elastic constituents
C = cij_from_group(116.3, 64.8, 30.9, group='m-3m') # GPa

# - 4. class instances
lattice = Lattice.from_matrix( M ) 
dislocation = Dislocation(lattice=lattice, hkl=hkl, uvw=uvw, phi=phi, SGno=None)
stroh = Stroh(C) # captures characteristic elastic matrix and eigensolution
calc = MLS(dislocation=dislocation, cij=C) # captures sum computation

# - 5. viz
fig, ax = dislocation.visualize()


# alias
D = dislocation

# - direct calc
h, k, l = 1, 1, 1
x1, x2, x3 = M
a, b, c = np.diag(M)
d111 = [
        np.sqrt(3)/3 * a,
        lattice.length((h, k, l)) / (h**2 + k**2 + l **2)
        ]

# - reciprocal calc
q111 = [
        2 * np.pi / (np.sqrt(3)/3 * a),
        2 * np.pi * lattice.reciprocal.length((1,1,1))
        ]



# - 6.
# In Martinez-Garcia et al. (2008) Phil Mag Lett 88(6) 443-451 the authors give
# the simplified form of the solution to the fundamental elasticity matrix `N`
# for the particular case where the line vector `l` lies along a rotational 
# diad, and perpendicular to a second perpendicular axis. (c.f. eqn 6)
#
# The terms are expressed in the terms of stiffness tranformed to the local 
# coordinate reference frame, so may differ in absolute terms, but the values
# should correctly be zero / non-zero

def analytic_ortho_N(c11, c22, c33, c44, c55, c66, c12):
    a = c12/c22
    b = c12 * a - c11
    N = np.array((
        (0    , -1   , 0    , 1/c66, 0    , 0    ),
        (-a   , 0    , 0    , 0    , 1/c22, 0    ),
        (0    , 0    , 0    , 0    , 0    , 1/c44),
        (b    , 0    , 0    , 0    , -a   , 0    ),
        (0    , 0    , 0    , -1   , 0    , 0    ),
        (0    , 0    , -c55 , 0    , 0    , 0    )
    ))
    return N

N = analytic_ortho_N(*np.diag(C), C[0,1])

diffs_6 = stroh.N - N
test_value_6 = all(np.abs(diffs_6).ravel() <= 1e-06)

if not test_value_6:
    print(f'stroh.N is {"equal" if test_value_6 else "unequal"} to analytic N with the following diff:\n{diffs_6}')


# - 7.
# Gijkl is given differently in MLS (2007) and MLS (2008), but represents a 
# tensor containing products of dot products representing direction cosines
# of the stresses and strains in the system. These are indicated as unit
# vectors in (2008), though the formulae are written differently.

s = hkl
Gijkl = np.zeros((3,2,3,2))
for i in range(3):
    for j in range(2):
        for k in range(3):
            for l in range(2):
                Gijkl[(i,j,k,l)] = np.product((               
                    np.dot(s, D.e[i]),
                    np.dot(s, D.e[j]),
                    np.dot(s, D.e[k]),
                    np.dot(s, D.e[l])
                ))


Gijkl_v2 = np.zeros((3,2,3,2))
for i in range(3):
    for j in range(2):
        for k in range(3):
            for l in range(2):
                Gijkl_v2[(i,j,k,l)] = np.product((               
                    D.tau(s)[i],
                    D.tau(s)[j],
                    D.tau(s)[k],
                    D.tau(s)[l]
                ))

Gijkl_v3 = D.Gijmn(s)


diffs_7 = D.Gijmn(s) - Gijkl
test_value_7 = all(np.abs(diffs_7).ravel() <= 1e-06)

if not test_value_7:    
    print(f'dislocation.Gijmn is {"equal" if test_value_7 else "unequal"} \
          to analytic Gijmn with the following diff:\n{diffs_7}')
    
    for i in range(3):
        for j in range(2):
            for k in range(3):
                for l in range(2):
                    idx = (i,j,k,l)
                    print(idx, Gijkl[idx], Gijkl_v2[idx], Gijkl_v3[idx])
