# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 12:47:13 2022

@author: pmetz1
"""
# 3rd party
import numpy as np

# package
from pymls import Lattice, Dislocation, Stroh, MLS
from pymls.elastic import cij_from_group
from pymls.toolbox import abt


# - 1. crystal lattice
lattice_scalar = (3.3065,) * 3 + (90,) * 3

# - 2. slip system
hkl = np.array((1,1,0))  # BCC slip plane
uvw = np.array((-1,1,1)) # burgers vector
l   = np.cross(uvw, hkl) # defines edge dislocation
phi = abt(uvw, l, degrees=True) # 90 degrees == edge dislocation

# - 3. elastic constituents
C = cij_from_group(131.4, 98.2, 28.2, group='m-3m')

# - 4. class instances
L = Lattice.from_scalar( lattice_scalar )
D = Dislocation(lattice=L, hkl=hkl, uvw=uvw, phi=phi, SGno=None)
S = Stroh(C) # captures characteristic elastic matrix and eigensolution
I = MLS(dislocation=D, cij=C) # captures sum computation

# - 5. compute values
# Anzic
# b[-1,1,1]; n[1,1,0]; l[-1,1-2]; g[-1,1,1]
Canzic = 0.44387384
Cmls = I.Chkl(uvw)
print(f'Anzic: {Canzic:.6f}; this work: {Cmls:.6f}')
print(f'Differs by Canzic / Cmls == {Canzic / Cmls:.6f}')
