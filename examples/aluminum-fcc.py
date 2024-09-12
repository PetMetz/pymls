# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:51:30 2022

@author: pmetz1
""" 

# 3rd party
import numpy as np

# package
from pymls import Lattice, Dislocation, Stroh, MLS
from pymls.elastic import cij_from_group, cij_from_dict
from pymls.toolbox import abt
from pymls.symmetry import Symmetry as SO
import pymls.toolbox as tbx


# - 1. crystal lattice
lattice_scalar = (4.03,) * 3 + (90,) * 3

# - 2. slip system
hkl = np.array((1,-1,-1))  # FCC slip plane
uvw = np.array((1,1,0)) # burgers vector
l   = np.cross(uvw, hkl) # defines edge dislocation
phi =  90 # abt(uvw, l, degrees=True) # 90 degrees == edge dislocation
chi = abt(hkl, uvw, degrees=True)

# - 3. elastic constituents
C = cij_from_group(116.3, 64.8, 30.9, group='m-3m') # GPa

# - 4. class instances
L = Lattice.from_scalar( lattice_scalar )
D = Dislocation(lattice=L, hkl=hkl, uvw=uvw, phi=phi, SGno=None)
S = Stroh(C, ) # captures characteristic elastic matrix and eigensolution
I = MLS(dislocation=D, cij=C) # captures sum computation

# - 5. compute values
# Anizc
# b[1,-1,0]; n[1,1,1]; l[-1,-1,2]; g[1,-1,0]
Canzic = 0.51008962
Cmls = I.Chkl(uvw)
print(f'Anzic: {Canzic:.6f}; this work: {Cmls:.6f}')
print(f'Differs by Canzic / Cmls == {Canzic / Cmls:.6f}')

# plot
D.visualize()
I.plot_u()
I.plot_beta()


#%% Sym Eqs
# m-3m (215) https://it.iucr.org/Ac/ch2o3v0001/sgtable2o3o225/
#  1
#  2 || (x,0,0), (0,y,0)
#  3 || (x,x,x)
#  2 || (x,x,0)
# -1 || (0,0,0)
R2x00 = SO.rotation((1,0,0), 180)
R20y0 = SO.rotation((0,1,0), 180)
R3xxx = SO.rotation((1,1,1), 120)
R2xx0 = SO.rotation((1,1,0), 180)
Inv   = SO.inversion()
SOS   = (R2x00, R20y0, R3xxx, R2xx0, Inv) # set of generators
N     = (1, 1, 2, 1, 1) # number of times to operate

slip = [hkl, uvw]
for _ in range(4): # redundant
    for symOpp, nOpp in list(zip(SOS, N)):                # for symOpp in generator set
        for _ in range(nOpp):                             # do N times
            slip = np.asarray(slip).reshape((-1,3))       # ...
            slip = np.concatenate((slip, symOpp(slip)))   # append new symmetric elements
            slip = slip.reshape((-1,2,3))                 # ...
            slip = tbx.get_unique_pairs(slip)             # find unique pairs (elements)
# slip = slip.astype(int) # this is changing nominal 1 values to zeros for some reason.... =(
slip = np.round(slip, decimals=0).astype(int)
m = ~np.array([np.dot(*e) for e in slip], dtype=bool)



all_combinations = []
for uvw, hkl in slip[m]:
    dislocation = Dislocation(lattice=L, hkl=hkl, uvw=uvw, phi=phi, SGno=None)
    stroh = Stroh(C) # captures characteristic elastic matrix and eigensolution
    calc = MLS(dislocation=dislocation, cij=C) # captures sum computation
    all_combinations.append(calc)

s = (1,1,0)
mean_C = np.mean([e.Chkl(s) for e in all_combinations])
mean_E = np.mean([e.Eij for e in all_combinations], axis=0)
mean_P = np.mean([e.stroh.P for e in all_combinations], axis=0)
