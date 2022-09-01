# -*- coding: utf-8 -*-
"""
Example given in Martinez-Garcia (2009).

Reference
---------
1. Martinez-Garcia, J., Leoni, M., & Scardi, P. (2009). A general approach 
   for determining the diffraction contrast factor of straight-line
   dislocations. Acta Crystallographica Section A Foundations of
   Crystallography, 65(2), 109–119. https://doi.org/10.1107/S010876730804186X



Created on Mon Jun 27 14:20:20 2022

@author: pmetz1
"""
import numpy as np

from pymls import Lattice, MLSLattice, Martinez

#%% Forsterite example
c11 = 328.7 # GPa
c22 = 199.8
c33 = 235.5
c23 = 72.67
c13 = 68.35
c12 = 66.75
c44 = 66.78
c55 = 80.95
c66 = 80.571
cij = np.array((
    (c11, c12, c13, 0  , 0  , 0  ),
    (c12, c22, c23, 0  , 0  , 0  ),
    (c13, c23, c33, 0  , 0  , 0  ),
    (0  , 0  , 0  , c44, 0  , 0  ),
    (0  , 0  , 0  , 0  , c55, 0  ),
    (0  , 0  , 0  , 0  , 0  , c66)
    ))
SG = "Pbnm"
scalar = (4.775, 10.190, 5.978, 90, 90, 90)
M = Lattice.from_scalar(scalar).M
hkl = np.array((0,1,0))
uvw = np.array((0,0,1))
phi = 90
L = MLSLattice(M, hkl, uvw, phi)
MLS = Martinez(cij, L)
