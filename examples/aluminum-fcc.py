# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:51:30 2022

@author: pmetz1
"""

# 3rd party
import numpy as np

# package
from pymls import Lattice, Dislocation, Stroh, MLS
from pymls.elastic import cij_from_group
from pymls.toolbox import abt


# - 1. crystal lattice
lattice_scalar = (4.03,) * 3 + (90,) * 3

# - 2. slip system
hkl = np.array((1,1,1))  # BCC slip plane
uvw = np.array((1,1,-2)) # burgers vector
l   = np.cross(uvw, hkl) # defines edge dislocation
phi = abt(uvw, l, degrees=True) # 90 degrees == edge dislocation

# - 3. elastic constituents
C = cij_from_group(116.3, 64.8, 30.9, group='m-3m') # GPa

# - 4. class instances
L = Lattice.from_scalar( lattice_scalar )
D = Dislocation(lattice=L, hkl=hkl, uvw=uvw, phi=phi, SGno=None)
S = Stroh(C) # captures characteristic elastic matrix and eigensolution
I = MLS(dislocation=D, cij=C) # captures sum computation

# - 5. compute values
# Anizc
# b[1,1,-2]; n[1,1,1]; l[3,-3,0]; g[1,1,-2]
Canzic = 0.51053827
Cmls = I.Chkl(uvw)
print(f'Anzic: {Canzic:.6f}; this work: {Cmls:.6f}')
print(f'Differs by Canzic / Cmls == {Canzic / Cmls:.6f}')

# %%
