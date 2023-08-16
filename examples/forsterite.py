# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 12:47:54 2022

@author: pmetz1
"""
# 3rd party
import matplotlib.pyplot as plt
import numpy as np

# package
from pymls import Lattice, Dislocation, Stroh, MLS
from pymls.elastic import cij_from_group
from pymls.toolbox import abt

plt.close('all')

# - 1. crystal lattice
lattice_scalar = (4.775, 10.190, 5.978, 90, 90, 90)

# - 2. slip system
hkl = np.array((0,1,0)) # {HKL}
uvw = np.array((1,0,0)) # <uvw>
l   = np.array((0,0,1)) # np.cross(hkl, uvw)
phi = abt(uvw, l, degrees=True) # 90 degrees == edge dislocation
chi = abt(hkl, uvw, degrees=True)

# - 3. elastic constituents
C = cij_from_group(  # GPa
          328 , # c11
          69  , # c12
          69  , # c13
          200 , # c22
          73  , # c23
          235 , # c33
          66.7, # c44
          81.3, # c55
          80.9, # c66
          group='mmm'
          )

# - 4. class instances
L = Lattice.from_scalar( lattice_scalar )
D = Dislocation(lattice=L, hkl=hkl, uvw=uvw, phi=phi, SGno=None)
S = Stroh(C) # captures characteristic elastic matrix and eigensolution
I = MLS(dislocation=D, cij=C) # captures sum computation

# - 5. compute values
sss = np.array(( # c.f. Table 5, MLS (2009)
    (0,2,0),
    (1,1,0),
    (0,2,1),
    (1,0,1)
    ))
cbar = np.array(( # c.f. Table 5, MLS (2009)
    0.1340,
    0.4548,
    0.0449,
    0.1773
    ))

for s, c in zip(sss, cbar):
    Cmls = I.Chkl(s)
    print(f'MLS (2009): {c:.6f}; this work: {Cmls:.6f}')
    print(f'Differs by c / Cmls == {c / Cmls:.6f}')
    print('')

# plot
plt.close('all')
D.visualize()