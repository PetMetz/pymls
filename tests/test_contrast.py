# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:03:42 2022

@author: pmetz1
"""
# 3rd party
import numpy as np
import pytest

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
                rv[a,m,n] = np.angle(A[m,a] * D[a] * P[a] ** (n-1)) 
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
                A[i,j] = modP[i] / (2 * P[i].imag**2)
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
    for l in range(3):
        for n in range(2):
            for m in range(3):
                for k in range(3):
                    for j in range(2):
                        for i in range(3):
                            rv[i,j,k,m,n,l] = 2 * A[i,k] * A[m,l] * D[k] \
                                * D[l] * P[k]**(j-1) * P[l]**(n-1)
    B = mls.phi
    assert tbx.float_tol(rv,B)


# --- classes
