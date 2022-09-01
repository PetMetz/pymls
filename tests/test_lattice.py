# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:58:27 2022

@author: pmetz1
"""
# 3rd party
import numpy as np
from scipy.spatial.transform import Rotation

# package
from pymls.lattice import Lattice
from pymls.toolbox import float_tol, generate_hull


# local
from fixtures import cubic_lattice, hexagonal_lattice, triclinic_lattice
from fixtures import lattice_suite


# --- declare constants
_SMALL = 1e-12
a = 3
b = 3
c = 3
al = 90
be = 90
ga = 90
Sc = np.array((a, b, c, al, be, ga))
Mc = np.eye(3) * np.array((a, b, c))
Gc = np.eye(3) * np.array((a, b, c))**2


# --- define test functions
...    

# --- define test collections
class TestInstantiation:
    """ check constructors result in identical scalar lattice """
    def test_lattice_from_scalar(self):
        assert float_tol(Lattice.from_scalar(Sc).scalar, Sc)

    def test_lattice_from_matrix(self):
        assert float_tol(Lattice.from_matrix(Mc).scalar, Sc)

    def test_lattice_from_metric(self):
        assert float_tol(Lattice.from_metric(Gc).scalar, Sc)
        
    ...
        


class TestAlgebra:
    """ check operations involving metric tensor """
    
    L = Lattice.from_scalar(Sc)
    
    def test_volume(self):
        """ V == det(G) """
        hull = generate_hull(self.L.matrix)
        assert float_tol(hull.volume, self.L.volume)
    
    def test_length(self):
        """ l == v @ G @ v """
        x = np.array((0.5, 0.5, 0.5))
        a = self.L.a
        G = self.L.G
        l1 = np.sqrt(3)/2 * a
        l2 = np.sqrt(x @ G @ x)
        assert float_tol(l1, l2)
        
    def test_angle(self):
        """ theta_ab = arccos( a @ b / |a||b| ) """
        a = np.array((3, 0, 0))
        b = np.array((0, 3, 0))
        t1 = np.arccos(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) ) # radians
        t2 = self.L.angle((1,0,0), (0,1,0))
        assert float_tol(t1, t2)
        

    def test_reciprocal(self):
        """ G @ G* == I """
        L = self.L.G @ self.L.reciprocal.G
        R = np.eye(3)
        assert float_tol(L.ravel(), R.ravel())
        
    def test_linalg_rotation(self):
        """ M` == P @ M """
        t = 2*np.pi/3   # angle
        Rz = np.array(( # z-rotation
            (np.cos(t), -np.sin(t), 0),
            (np.sin(t),  np.cos(t), 0),
            (0        ,          0, 1)
            ))
        Mp   = Rz @ self.L.copy().M # definition
        Lp  = self.L.copy().transform(Rz) # class method 1 (linear algebra)
        assert float_tol(Mp.ravel(), Lp.M.ravel())

    def test_scipy_rotation(self):
        """ M` == P @ M """
        t = 2*np.pi/3   # angle
        Rz = np.array(( # z-rotation
            (np.cos(t), -np.sin(t), 0),
            (np.sin(t),  np.cos(t), 0),
            (0        ,          0, 1)
            ))
        Mp   = Rz @ self.L.copy().M # definition
        Lp = self.L.copy().transform(Rotation.from_euler('z', -t)) # class method 2 (scipy.spatial)
        assert float_tol(Mp.ravel(), Lp.M.ravel())
        
    ...