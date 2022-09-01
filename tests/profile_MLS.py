# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 12:39:18 2022

@author: pmetz1
"""
import numpy as np
from gist_contrast_factor import MLSLattice, Martinez


# 1. crystal details
C = np.eye(3) * 3.25
spacegroup = "Im-3m"
c11 = 131.4 # GPa
c44 = 28.8 # GPa
c12 = 98.2 # GPa
dia = (c11,)*3 + (c44,)*3
cij = np.eye(6) * dia
cij[0,1] = cij[1,0] = cij[0,2] = cij[2,0] = cij[1,2] = cij[2,1] = c12

# 2. slip system details
o   = np.array((0,0,0)) # origin
hkl = np.array((1,1,0))  # BCC slip plane
uvw = np.array((-1,1,1)) # burgers vector
l   = np.cross(uvw, hkl) # defines edge dislocation
phi = 90 #  abt(uvw, l, degrees=True)

# 3. constituents
L = MLSLattice(C, hkl, uvw, phi) # captures geometric aspects
# s = Stroh(cij) # captures elastic eigenproblem
# _ = L.Gijmn(uvw)

# 4. interface
MLS = Martinez(cij, L) # captures sum computation

# 5. compute
# _ = MLS.Eijmn
_ = MLS.Chkl((-1,1,0))
