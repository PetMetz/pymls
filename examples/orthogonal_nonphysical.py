# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 16:27:19 2022

@author: UT
"""

# 3rd party
import numpy as np
from numpy import linalg as LA

# package
from pymls import Lattice, Dislocation, Stroh, MLS
from pymls.elastic import cij_from_group
from pymls.toolbox import abt


# - 1. crystal lattice
M = 2 * np.eye(3)

# - 2. slip system
hkl = np.array((0,0,1))  # BCC slip plane
uvw = np.array((1,0,0))  # burgers vector
l   = np.array((0,1,0))  # dislocation line vector
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
