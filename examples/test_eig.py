# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 18:24:47 2023

@author: UT
"""

import numpy as np
import scipy.linalg as sla

from pymls import toolbox as tbx


def get_norm(a, b):
    return 

A = np.arange(9).reshape((3,3))
w, vl, vr = sla.eig(A, left=True, right=True)

B = np.random.uniform(size=36).reshape((6,6))
w2, vl2, vr2 = sla.eig(B, left=True, right=True)


#%%  eta_a dot xi_b = delta_ab
ans = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        ans[i,j] = vl[:,i] @ vr[:,j]

mat = vl.T @ vr
print(tbx.float_tol(ans,mat))


#%%  b.T a + a.T b = delta_ab
ans = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        ans[i,j] = vr[:,i] @ vl[:,j] + vl[:,i] @ vr[:,j]

N = np.diag(ans)
print(N)
print((vl/N).T @ vr/N)







