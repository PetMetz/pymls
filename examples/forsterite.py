# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 12:47:54 2022

@author: pmetz1
"""
# 3rd party
import numpy as np

# package
from pymls import Lattice, Dislocation, Stroh, MLS
from pymls.elastic import cij_from_group
from pymls.toolbox import abt


# - 1. crystal lattice
lattice_scalar = (4.775, 10.190, 5.978, 90, 90, 90)

# - 2. slip system
hkl = np.array((0,1,0))
uvw = np.array((0,0,1))
l   = np.cross(hkl, uvw)
phi = 90

# - 3. elastic constituents
C = cij_from_group(  # GPa
          328.7, # c11
          199.8, # c22
          235.5, # c33
          72.67, # c23
          68.35, # c13
          66.75, # c12
          66.78, # c44
          80.95, # c55
          80.57, # c66
          group='mmm')

# - 4. class instances
L = Lattice.from_scalar( lattice_scalar )
D = Dislocation(lattice=L, hkl=hkl, uvw=uvw, phi=phi, SGno=None)
S = Stroh(C) # captures characteristic elastic matrix and eigensolution
I = MLS(dislocation=D, cij=C) # captures sum computation

# - 5. compute values
Cmls = I.Chkl(uvw)
