# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 18:23:13 2022

@author: UT
"""

# 3rd party
import numpy as np
import numpy.linalg as LA
import pytest

# package
from pymls import toolbox as tbx
from pymls.elastic import cij_from_group


# ---
def fn(X):
    return 0.5 * (X[(0,0)] - X[(0,1)]) # NB zero index


# ---
groups = ['-1', '2/m', 'mmm', '4/m', '4/mmm', '-3', '-3m', '6/m', '6/mmm', 'm-3', 'm-3m']
orders = [
    [11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 33, 34, 35, 36, 44, 45, 46, 55, 56, 66],
    [11, 12, 13, 15, 22, 23, 25, 33, 35, 44, 46, 55, 66],
    [11, 12, 13, 22, 23, 33, 44, 55, 66],
    [11, 12, 13, 16, 33, 44, 66],
    [11, 12, 13, 33, 44, 66],
    [11, 12, 13, 14, 15, 33, 44],
    [11, 12, 13, 14, 33, 44],
    [11, 12, 13, 33, 44],
    [11, 12, 13, 33, 44],
    [11, 12, 44],
    [11, 12, 44]
    ]


def test_1():
    A = np.array((
        (11, 12, 13, 14, 15, 16),
        ( 0, 22, 23, 24, 25, 26),
        ( 0,  0, 33, 34, 35, 36),
        ( 0,  0,  0, 44, 45, 46),
        ( 0,  0,  0,  0, 55, 56),
        ( 0,  0,  0,  0,  0, 66)
        ), dtype=float)
    A = A + A.T - np.diag(A.diagonal())
    B = cij_from_group(*np.unique(orders[1-1]), group='-1')
    assert tbx.float_tol(A, B)


def test_2():
    A = np.array((
        (11, 12, 13,  0, 15,  0),
        ( 0, 22, 23,  0, 25,  0),
        ( 0,  0, 33,  0, 35,  0),
        ( 0,  0,  0, 44,  0, 46),
        ( 0,  0,  0,  0, 55,  0),
        ( 0,  0,  0,  0,  0, 66)
        ), dtype=float)
    A = A + A.T - np.diag(A.diagonal())
    B = cij_from_group(*np.unique(orders[2-1]), group='2/m')
    assert tbx.float_tol(A, B)


def test_3():
    A = np.array((
        (11, 12, 13,  0,  0,  0),
        ( 0, 22, 23,  0,  0,  0),
        ( 0,  0, 33,  0,  0,  0),
        ( 0,  0,  0, 44,  0,  0),
        ( 0,  0,  0,  0, 55,  0),
        ( 0,  0,  0,  0,  0, 66)
        ), dtype=float)
    A = A + A.T - np.diag(A.diagonal())
    B = cij_from_group(*np.unique(orders[3-1]), group='mmm')
    assert tbx.float_tol(A, B)


def test_4():
    A = np.array((
        (11, 12, 13,  0,  0, 16),
        ( 0, 11, 13,  0,  0,-16),
        ( 0,  0, 33,  0,  0,  0),
        ( 0,  0,  0, 44,  0,  0),
        ( 0,  0,  0,  0, 44,  0),
        ( 0,  0,  0,  0,  0, 66)
        ), dtype=float)
    A = A + A.T - np.diag(A.diagonal())
    B = cij_from_group(*np.unique(orders[4-1]), group='4/m')
    assert tbx.float_tol(A, B)


def test_5():
    A = np.array((
        (11, 12, 13,  0,  0,  0),
        ( 0, 11, 13,  0,  0,  0),
        ( 0,  0, 33,  0,  0,  0),
        ( 0,  0,  0, 44,  0,  0),
        ( 0,  0,  0,  0, 44,  0),
        ( 0,  0,  0,  0,  0, 66)
        ), dtype=float)
    A = A + A.T - np.diag(A.diagonal())
    B = cij_from_group(*np.unique(orders[5-1]), group='4/mmm')
    assert tbx.float_tol(A, B)


def test_6():
    A = np.array((
        (11, 12, 13, 14, 15,  0),
        ( 0, 11, 13,-14,-15,  0),
        ( 0,  0, 33,  0,  0,  0),
        ( 0,  0,  0, 44,  0,-15),
        ( 0,  0,  0,  0, 44, 14),
        ( 0,  0,  0,  0,  0, 99)
        ), dtype=float)
    A = A + A.T - np.diag(A.diagonal())
    A[-1,-1] = fn(A)
    B = cij_from_group(*np.unique(orders[6-1]), group='-3')
    assert tbx.float_tol(A, B)


def test_7():
    A = np.array((
        (11, 12, 13, 14,  0,  0),
        ( 0, 11, 13,-14,  0,  0),
        ( 0,  0, 33,  0,  0,  0),
        ( 0,  0,  0, 44,  0,  0),
        ( 0,  0,  0,  0, 44, 14),
        ( 0,  0,  0,  0,  0, 99)
        ), dtype=float)
    A = A + A.T - np.diag(A.diagonal())
    A[-1,-1] = fn(A)
    B = cij_from_group(*np.unique(orders[7-1]), group='-3m')
    assert tbx.float_tol(A, B)


def test_8():
    A = np.array((
        (11, 12, 13,  0,  0,  0),
        ( 0, 11, 13,  0,  0,  0),
        ( 0,  0, 33,  0,  0,  0),
        ( 0,  0,  0, 44,  0,  0),
        ( 0,  0,  0,  0, 44,  0),
        ( 0,  0,  0,  0,  0, 99)
        ), dtype=float)
    A = A + A.T - np.diag(A.diagonal())
    A[-1,-1] = fn(A)
    B = cij_from_group(*np.unique(orders[8-1]), group='6/m')
    assert tbx.float_tol(A, B)


def test_9():
    A = np.array((
        (11, 12, 13,  0,  0,  0),
        ( 0, 11, 13,  0,  0,  0),
        ( 0,  0, 33,  0,  0,  0),
        ( 0,  0,  0, 44,  0,  0),
        ( 0,  0,  0,  0, 44,  0),
        ( 0,  0,  0,  0,  0, 99)
        ), dtype=float)
    A = A + A.T - np.diag(A.diagonal())
    A[-1,-1] = fn(A)
    B = cij_from_group(*np.unique(orders[9-1]), group='6/mmm')
    assert tbx.float_tol(A, B)


def test_10():
    A = np.array((
        (11, 12, 12,  0,  0,  0),
        ( 0, 11, 12,  0,  0,  0),
        ( 0,  0, 11,  0,  0,  0),
        ( 0,  0,  0, 44,  0,  0),
        ( 0,  0,  0,  0, 44,  0),
        ( 0,  0,  0,  0,  0, 44)
        ), dtype=float)
    A = A + A.T - np.diag(A.diagonal())
    B = cij_from_group(*np.unique(orders[10-1]), group='m-3')
    assert tbx.float_tol(A, B)


def test_11():
    A = np.array((
        (11, 12, 12,  0,  0,  0),
        ( 0, 11, 12,  0,  0,  0),
        ( 0,  0, 11,  0,  0,  0),
        ( 0,  0,  0, 44,  0,  0),
        ( 0,  0,  0,  0, 44,  0),
        ( 0,  0,  0,  0,  0, 44)
        ), dtype=float)
    A = A + A.T - np.diag(A.diagonal())
    B = cij_from_group(*np.unique(orders[11-1]), group='m-3m')
    assert tbx.float_tol(A, B)


#    1     2      3      4      5        6     7      8      9        10     11
#  ['-1', '2/m', 'mmm', '4/m', '4/mmm', '-3', '-3m', '6/m', '6/mmm', 'm-3', 'm-3m']
_ELASTIC_RESTRICTIONS = np.array((
  # (1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11), # Laue group
    (11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11),
    (12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12),
    (13, 13, 13, 13, 13, 13, 13, 13, 13, 12, 12),
    (14, 0 , 0 , 0 , 0 , 14, 14, 0 , 0 , 0 , 0 ),
    (15, 15, 0 , 0 , 0 , 15, 0 , 0 , 0 , 0 , 0 ),
    (16, 0 , 0 , 16, 0 , 0 , 0 , 0 , 0 , 0 , 0 ),
    (22, 22, 22, 11, 11, 11, 11, 11, 11, 11, 11),
    (23, 23, 23, 13, 13, 13, 13, 13, 13, 12, 12),
    (24, 0 , 0 , 0 , 0 ,-14,-14, 0 , 0 , 0 , 0 ),
    (25, 25, 0 , 0 , 0 ,-15, 0 , 0 , 0 , 0 , 0 ),
    (26, 0 , 0 ,-16, 0 , 0 , 0 , 0 , 0 , 0 , 0 ),
    (33, 33, 33, 33, 33, 33, 33, 33, 33, 11, 11),
    (34, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ),
    (35, 35, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ),
    (36, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ),
    (44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44),
    (45, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ),
    (46, 46, 0 , 0 , 0 ,-15, 0 , 0 , 0 , 0 , 0 ),
    (55, 55, 55, 44, 44, 44, 44, 44, 44, 44, 44),
    (56, 0 , 0 , 0 , 0 , 14, 14, 0 , 0 , 0 , 0 ),
    (66, 66, 66, 66, 66, 99, 99, 99, 99, 44, 44)
    ), dtype=int)