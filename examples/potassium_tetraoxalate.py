# -*- coding: utf-8 -*-
"""
Acta Cryst. (1964). 17, 1511
"The Crystal Structure of Potassium Tetraoxalate, K(HC204)(H2C204). 2H20"
KH3(C204)2.2 H20
SpaceGroup: P1 or P-1
LatticeConstants: (7.04, 10.59, 6.35, 101.4, 100.2, 94.0)
Density: 1.85 g cm**-3
Z: 2

Created on Sat Aug 12 14:47:59 2023

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
uvw = np.array((1,1,1))  # burgers vector
l   = np.cross(hkl, uvw)  # dislocation line vector
phi = abt(uvw, l, degrees=True) # 90 degrees == edge dislocation

# - 3. elastic constituents
def triclinic_cij():
    """
    Acta Cryst. (1970). A26, 401 
    "Triclinic Crystals Ammonium and Potassium Tetroxalate Dihydrate"
    KH3(C204)2.2 H20
    """
    cij = { # x 10^11 dyne cm**-2
           11: 2.536,
           12: 1.184,
           13: 0.983,
           14: 0.072,
           15: 0.612,
           16:-0.123,
           22: 4.779,
           23: 1.402,
           24: 1.134,
           25: 0.146,
           26:-0.270,
           33: 3.430,
           34: 0.219,
           35: 0.147,
           36: 0.040,
           44: 1.019,
           45:-0.082,
           46: 0.053,
           55: 0.569,
           56: 0.070,
           66: 0.499
           } # dicts are ordered in python 3.x
    GPa = 100 * np.array(list(cij.values())) # 10^11 dyne cm**-2 -> 100 GPa
    return cij_from_group(*GPa, group='-1')

C = triclinic_cij()

# - 4. class instances
L = Lattice.from_scalar((7.04, 10.59, 6.35, 101.4, 100.2, 94.0))
D = Dislocation(lattice=L, hkl=hkl, uvw=uvw, phi=phi, SGno=None)
S = Stroh(C) # captures characteristic elastic matrix and eigensolution
I = MLS(dislocation=D, cij=C) # captures sum computation

# - 5. viz
fig, ax = D.visualize()

Cmls = I.Chkl(uvw)
print(Cmls)