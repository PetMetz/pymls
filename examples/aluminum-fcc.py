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
uvw = np.array((1,1,0)) # burgers vector
l   = np.cross(uvw, hkl) # defines edge dislocation
phi = abt(uvw, l, degrees=True) # 90 degrees == edge dislocation

# - 3. elastic constituents
C = cij_from_group(116.3, 64.8, 30.9, group='m-3m') # GPa

# - 4. class instances
lattice = Lattice.from_scalar( lattice_scalar )
dislocation = Dislocation(lattice=lattice, hkl=hkl, uvw=uvw, phi=phi, SGno=None)
stroh = Stroh(C) # captures characteristic elastic matrix and eigensolution
calc = MLS(dislocation=dislocation, cij=C) # captures sum computation

# - 5. compute values
# Anizc
# b[1,1,0]; n[1,1,1]; l[1,-1,0]; g[1,1,0]
Canzic = 0.55027571
Cmls = calc.Chkl(uvw)
print(f'Anzic: {Canzic:.6f}; this work: {Cmls:.6f}')
print(f'Differs by Canzic / Cmls == {Canzic / Cmls:.6f}')


# %%
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

M = MLS_M(lattice)
L = Lattice(M)
D = Dislocation(lattice=L, hkl=hkl, uvw=uvw, phi=phi)
C2 = MLS(D, C)
Cmls2 = C2.Chkl(uvw)

