# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 14:18:21 2023

@author: pmetz1
"""
import numpy as np
import spglib as sg


# package
from pymls import Lattice, Dislocation, Stroh, MLS
from pymls.elastic import cij_from_group
from pymls.toolbox import abt


# - 1. crystal lattice
lattice_scalar = (4.03,) * 3 + (90,) * 3

# - 2. slip system
hkl = np.array((1,1,1))  # FCC slip plane
uvw = np.array((1,1,0)) # burgers vector
l   = np.cross(uvw, hkl) # defines edge dislocation
phi =  90 # abt(uvw, l, degrees=True) # 90 degrees == edge dislocation
chi = abt(hkl, uvw, degrees=True)

# - 3. elastic constituents
C = cij_from_group(116.3, 64.8, 30.9, group='m-3m') # GPa

# - 4. class instances
L = Lattice.from_scalar( lattice_scalar )
S = Stroh(C, ) # captures characteristic elastic matrix and eigensolution
# D = Dislocation(lattice=L, hkl=hkl, uvw=uvw, phi=phi, SGno=None)
# I = MLS(dislocation=D, cij=C) # captures sum computation




#%% get a symmetry generator

sym = sg.get_symmetry_from_database(523)

hkluvw = np.array((hkl,uvw))
complete = np.empty((len(sym['rotations']), len(hkl), 3), dtype=int)
for ii, o in enumerate(sym['rotations']):
    complete[ii] = (o @ hkl.T).T

un = np.unique(complete,axis=0)


#%%
I = []
for hkl, uvw in un:
    D = Dislocation(lattice=L, hkl=hkl, uvw=uvw, phi=phi, SGno=None)
    I.append( MLS(dislocation=D, cij=C) )
    
    
