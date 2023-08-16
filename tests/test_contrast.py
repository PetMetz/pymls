# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:03:42 2022

@author: pmetz1
"""
# 3rd party
import numpy as np
import pytest
import functools

# package
from pymls.contrast import MLS
from pymls import toolbox as tbx

# local
from fixtures import *
from fixtures import mls_suite


# --- constants

# --- fixtures

# --- functions
@pytest.mark.parametrize('thisFixture', mls_suite)
def test_y_algebra(thisFixture, request):
    """ compact expression yields correct result """
    mls = request.getfixturevalue(thisFixture)
    P = mls.stroh.P
    A = np.empty((3,3))
    for j in range(3):
        for i in range(3):
            A[i,j] = np.arctan((P[i].real - P[j].real) / (P[i].imag + P[j].imag))
    B = mls.y
    assert tbx.float_tol(A,B)


@pytest.mark.parametrize('thisFixture', mls_suite)
def test_y_skew_symmetric(thisFixture, request):
    """ compact expression yields correct result """
    mls = request.getfixturevalue(thisFixture)
    assert tbx.is_skew_symmetric(mls.y)


@pytest.mark.parametrize('thisFixture', mls_suite)
def test_F_algebra(thisFixture, request):
    """ compact expression yields correct result """
    mls = request.getfixturevalue(thisFixture)
    P = mls.stroh.P
    A = np.empty((3,3))
    for j in range(3):
        for i in range(3):
            A[i,j] = (P[i].real - P[j].real)**2 + (P[i].imag - P[j].imag)**2
    B = mls.F
    assert tbx.float_tol(A,B)


@pytest.mark.parametrize('thisFixture', mls_suite)
def test_F_symmetric(thisFixture, request):
    """ compact expression yields correct result """
    mls = request.getfixturevalue(thisFixture)
    assert tbx.is_symmetric(mls.F)


@pytest.mark.parametrize('thisFixture', mls_suite)
def test_Q_algebra(thisFixture, request):
    """ compact expression yields correct result """
    mls = request.getfixturevalue(thisFixture)
    P = mls.stroh.P
    modP = np.abs(P)**2 # (p * np.conjugate(p).T).real 
    A = np.empty((3,3), dtype=float)
    for j in range(3):
        for i in range(3):
            A[i,j] = (modP[i] - modP[j])**2 + \
                     4 * (P[i].real - P[j].real) * (modP[j] * P[i].real - modP[i] * P[j].real)
    B = mls.Q
    assert tbx.float_tol(A,B)


@pytest.mark.parametrize('thisFixture', mls_suite)
def test_Q_symmetric(thisFixture, request):
    """ compact expression yields correct result """
    mls = request.getfixturevalue(thisFixture)
    assert tbx.is_symmetric(mls.Q)
    

@pytest.mark.parametrize('thisFixture', mls_suite)
def test_gamma1_algebra(thisFixture, request):
    """ compact expression yields correct result """
    mls = request.getfixturevalue(thisFixture)
    P = mls.stroh.P
    x = mls.x
    A = np.empty((3,3), dtype=float)
    for j in range(3):
        for i in range(3):
            A[i,j] = P[i].imag * P[j].imag * (np.tan(x[i]) + np.tan(x[j]))
    B = mls.gamma1
    assert tbx.float_tol(A,B)


@pytest.mark.parametrize('thisFixture', mls_suite)
def test_gamma1_symmetric(thisFixture, request):
    """ compact expression yields correct result """
    mls = request.getfixturevalue(thisFixture)
    assert tbx.is_symmetric(mls.gamma1)
    

@pytest.mark.parametrize('thisFixture', mls_suite)
def test_gamma2_algebra(thisFixture, request):
    """ compact expression yields correct result """
    mls = request.getfixturevalue(thisFixture)
    P = mls.stroh.P
    modP = np.abs(P)**2
    A = np.empty((3,3), dtype=float)
    for j in range(3):
        for i in range(3):
            A[i,j] = modP[i] - (P[i].real * P[j].real) + (P[i].imag * P[j].imag)
    B = mls.gamma2
    assert tbx.float_tol(A,B)
   
    
@pytest.mark.parametrize('thisFixture', mls_suite)
def test_delta_algebra(thisFixture, request):
    """ compact expression yields correct result; (a,m,n) -> (3,3,2) radians """
    mls = request.getfixturevalue(thisFixture)
    P = mls.stroh.P
    D = mls.D
    A = mls.stroh.A
    rv = np.empty((3,3,2), dtype=float)
    for n in range(2):
        for m in range(3):
            for a in range(3):
                rv[a,m,n] = np.angle(A[m,a] * D[a] * P[a] ** (n+1-1)) 
    B = mls.delta
    assert tbx.float_tol(rv,B)


@pytest.mark.parametrize('thisFixture', mls_suite)
def test_psi_algebra(thisFixture, request):
    """ compact expression yields correct result """
    mls = request.getfixturevalue(thisFixture)
    P = mls.stroh.P
    modP = np.abs(P)
    F = mls.F
    Q = mls.Q
    A = np.empty((3,3), dtype=float)
    for j in range(3):
        for i in range(3):
            if i == j:
                A[i,i] = modP[i] / (2 * P[i].imag**2)
            elif i != j:
                A[i,j] = (modP[i] / P[i].imag) * (F[i,j] / Q[i,j]) ** 0.5
    B = mls.psi
    assert tbx.float_tol(A,B)


@pytest.mark.parametrize('thisFixture', mls_suite)
def test_phi_algebra(thisFixture, request):
    """
    compact expression yields correct result;
    (i,j,a,m,n,a`) == 
    (i,j,k,m,n,l ) ==
    (3,2,3,3,2,3 ) 
    """
    mls = request.getfixturevalue(thisFixture)
    A = np.abs(mls.stroh.A)
    D = np.abs(mls.D)
    P = np.abs(mls.stroh.P)
    rv = np.empty((3,2,3,3,2,3), dtype=float)
    for l in range(3): # eig (col)
        for n in range(2): # exponent
            for m in range(3):
                for k in range(3): # eig (col) 
                    for j in range(2): # exponent
                        for i in range(3):
                            rv[i,j,k,m,n,l] = 2 * A[i,k] * A[m,l] * D[k] \
                                * D[l] * P[k]**int(j+1-1) * P[l]**int(n+1-1)
    B = mls.phi
    assert tbx.float_tol(rv,B)


# --- classes

@pytest.mark.parametrize('thisFixture', mls_suite)
class TestEijmnContraction:
    """ c.f. MLS (2009) eqn. 17 """

    @pytest.fixture(autouse=True)
    def _mls_class_fixture(self, thisFixture, request):
        """ alias """
        self.mls = request.getfixturevalue(thisFixture)
        self.delta = self.mls.delta
        self.x = self.mls.x
        self.y = self.mls.y
        self.z = self.mls.z
        self.phi = self.mls.phi
        self.psi = self.mls.psi

    @functools.cached_property
    def get_c1(self):
        A = np.zeros((3,3,2))
        for n in range(2): # exponent
            for m in range(3):
                for a in range(3):
                    A[a,m,n] = self.delta[a,m,n] + self.x[a]
        return np.cos(A)
    
    @functools.cached_property
    def get_c2(self):
        A = np.zeros((3,3,3,2))
        for j in range(2): # exponent
            for i in range(3):
                for b in range(3):
                    for a in range(3):
                        A[a,b,i,j] = self.delta[b,i,j] - self.y[a,b]
        return np.cos(A)
    
    @functools.cached_property
    def get_s1(self):
        A = np.zeros((3,3,2))
        for n in range(2): # exponent
            for m in range(3):
                for a in range(3):
                    A[a,m,n] = self.delta[a,m,n]
        return np.sin(A)
    
    @functools.cached_property
    def get_s2(self):
        A = np.zeros((3,3,3,2))
        for j in range(2): # exponent
            for i in range(3):
                for b in range(3):
                    for a in range(3):
                        A[a,b,i,j] = self.delta[b,i,j] + self.z[a,b]
        return np.sin(A)
    
    @functools.cached_property
    def get_direction_matrix(self):
        A = np.zeros((3,2,3,3,2,3)) # i, j, a, m, n, b
        c1 = self.get_c1
        c2 = self.get_c2
        s1 = self.get_s1
        s2 = self.get_s2
        for b in range(3): # eig
            for n in range(2): # exponent
                for m in range(3):
                    for a in range(3): # eig
                        for j in range(2): # exponent
                            for i in range(3):
                                A[i,j,a,m,n,b] = c1[a,m,n] * c2[a,b,i,j] + s1[a,m,n] * s2[a,b,i,j]
        return A
    
    def test_c1(self):
        r""" :math:`\cos(\Delta_{\alpha}^{mn} + x_{\alpha})` """ 
        # return np.cos( np.einsum('amn,a->amn', self.delta, self.x) )
        A = self.get_c1
        B = self.mls._c1
        assert tbx.float_tol(A,B)
        
    def test_c2(self) -> np.ndarray:
        r""" :math:`\cos(\Delta_{ij}^{\alpha'} - y_{\alpha}^{\alpha'})` """
        # return np.cos( np.einsum('bmn,ab->amn', self.delta, -self.y) )
        A = self.get_c2
        B = self.mls._c2
        assert tbx.float_tol(A,B)
    
    def test_s1(self) -> np.ndarray:
        r""" :math:`\sin(\Delta_{\alpha}^{mn})` """
        # return np.sin(self.delta)
        A = self.get_s1
        B = self.mls._s1
        assert tbx.float_tol(A,B)
    
    def test_s2(self) -> np.ndarray:
        r""" :math:`\sin(\Delta_{ij}^{\alpha'} - z_{\alpha}^{\alpha'})` """
        # return np.sin( np.einsum('bmn,ab->amn', self.delta, self.z) )
        A = self.get_s2
        B = self.mls._s2
        assert tbx.float_tol(A,B)
    
    def test_direction_matrix(self):
        r""" [_c1 * _c2 + _s1 * _s2] """
        # C = np.einsum('amn,bij->ijamnb', self._c1, self._c2) + np.einsum('amn,bij->ijamnb', self._s1, self._s2) # (i,j,a,m,n,b) == (3,2,3,3,2,3)
        c1 = self.get_c1
        c2 = self.get_c2
        s1 = self.get_s1
        s2 = self.get_s2
        A = self.get_direction_matrix
        B = np.einsum('amn,abij->ijamnb', c1, c2) + np.einsum('amn,abij->ijamnb', s1, s2)
        assert tbx.float_tol(A,B)
        
    def test_eijmn_contraction(self):
        r""" sum_ab [ psi_ab phi_ija^mnb dm_ija^mnb ] """
        A = np.zeros((3,2,3,2)) # i, j, m, n
        D = self.get_direction_matrix
        for n in range(2):
            for m in range(3):
                for j in range(2):
                    for i in range(3):
                        for b in range(3): # contract
                            for a in range(3): # contract
                                A[i,j,m,n] = self.psi[a,b] * self.phi[i,j,a,m,n,b] * D[i,j,a,m,n,b] + A[i,j,m,n]
        B = self.mls.Eijmn
        assert tbx.float_tol(A,B)
