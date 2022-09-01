# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 12:49:22 2022

@author: pmetz1
"""
# 3rd party
import numpy as np

# package
from pymls import Lattice, Dislocation, Stroh, MLS
from pymls.elastic import cij_from_group
from pymls.toolbox import abt


# 1. crystal lattice
lattice_scalar = (2.95, 2.95, 4.69, 90, 90, 120)

# 2. slip system
hkl = np.array((0,0,2))  # HCP basal slip
uvw = np.array((1,1,0)) # burgers vector
l   = np.cross(uvw, hkl) # defines edge dislocation
phi = abt(uvw, l, degrees=True) # 90 degrees == edge dislocation

# 3. elastic constituents
C = cij_from_group(135.6, 71.1, 24.1, 145.7, 45.7, group='6/mmm')

# 4. class instances
L = Lattice.from_scalar( lattice_scalar )
D = Dislocation(lattice=L, hkl=hkl, uvw=uvw, phi=phi, SGno=None)
S = Stroh(C) # captures characteristic elastic matrix and eigensolution
I = MLS(dislocation=D, cij=C) # captures sum computation

# 5. compute
# Anzic
# b[1,1,-2,0]; n[0,0,0,2]; l[2,-2,0,0]; g[1,1,-2,0]
Canzic = 0.54240600
Cmls = I.Chkl(uvw)
print(f'Anzic: {Canzic}; this work: {Cmls}')
print(f'Differs by Canzic / Cmls == {Canzic / Cmls:.6f}')
