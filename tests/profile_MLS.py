# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 12:39:18 2022

@author: pmetz1
"""
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

# 5. compute
# _ = MLS.Eijmn
rv = I.Chkl((-1,1,0)) # compute contrast factor
