# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:02:50 2022

@author: pmetz1
"""
# built-ins
from __future__ import annotations
from typing import Iterable

# 3rd party
import numpy as np
import scipy as sp
from numpy import linalg as LA

# package
from . import toolbox as tbx


class Lattice():
    """ simple representation of a lattice """
    
    # - state variables
    _G = None
    _reciprocal = None
    
    def __repr__(self):
        return f'<Lattice(a={self.a:.5f}, b={self.b:.5f}, c={self.c:.5f}, alpha={self.al:.5f}, beta={self.be:.5f}, gamma={self.ga:.5f} @ {hex(id(self))} >'
                     
    def __hash__(self):
        """ overload built in hash """
        return hash((self.a, self.b, self.c, self.al, self.be, self.ga))

    def __eq__(self, other):
        """ overload built-in equal """
        if isinstance(other, Lattice):
            return self.__hash__() == other.__hash__()
        return NotImplemented

    def __ne__(self, other):
        """ overload built-in not equal """
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __init__(self, matrix:np.ndarray=None) -> None:
        """ intialize unit cell parameters ($\AA$ and $\degree$)"""
        self.matrix = matrix
        
    # - constructors
    @classmethod
    def from_scalar(cls, x:tuple) -> Lattice:
        """ Julian 2014 Foundations of Crystallography """
        x = np.asarray(x)
        a, b, c = x[:3] # unpack
        angles = x[3:] * np.pi / 180
        cos = np.cos(angles)
        sin = np.sin(angles)
        c1 = c * cos[1]
        c2 = c * (cos[0] - cos[2]*cos[1]) / sin[2]
        c3 = np.sqrt(c**2 - c1**2 - c2**2)
        X  = np.array((
                      (a         , 0         , 0 ),
                      (b * cos[2], b * sin[2], 0 ),
                      (c1        , c2        , c3)
                      ))
        return cls(X)
        
    @classmethod
    def from_matrix(cls, x:np.ndarray) -> Lattice:
        """ """
        return cls(x)

    # FIXME there must be a linear algebra route to this.
    @classmethod
    def from_metric(cls, x:np.ndarray) -> Lattice:
        r"""
        .. math::
            
            \vec{a}\cdot\vec{b} = a b cos(\theta)
            
            \theta = cos^{-1}\left(\frac{\vec{a}\cdot\vec{b}}{a b}\right)
            
        """
        abc = np.sqrt(np.diag(x))
        angles = np.array((
            np.arccos(x[1,2] / np.product(abc[[1,2]])), # arccos(bc cos(alpha) / bc)
            np.arccos(x[0,2] / np.product(abc[[0,2]])), # arccos(ac cos(beta)  / ac)
            np.arccos(x[0,1] / np.product(abc[[0,1]]))  # arccos(ab cos(gamma) / ab)
            )) * 180 / np.pi
        return cls.from_scalar(np.concatenate((abc, angles)))
    
    @classmethod
    def dispatch_constructor(cls, x) -> Lattice:
        """ """
        if isinstance(x, Lattice): # don't know why you'd want this
            return x.copy()
        elif isinstance(x, Iterable) and len(x) == 6: # scalar
            return cls.from_scalar(x)
        elif isinstance(x, np.ndarray) and tbx.float_tol(x, x.T): # symmetric matrix == metric matrix
            return cls.from_metric(x)
        elif isinstance(x, np.ndarray) and not tbx.float_tol(x, x.T): # asymmetric matrix == vector representation
            return cls.from_matrix(x)
        else:
            raise Exception(f'Unable to dispatch lattice constructor: {x}')
    
    # - mutations
    def copy(self) ->Lattice:
        """ return new instance of self """
        return Lattice(self.matrix) # deepcopy(self)
    
    def transform(self, x:(np.ndarray,sp.spatial.transform.Rotation)) -> Lattice:
        # - as matrix
        if isinstance(x, np.ndarray):
            assert x.shape == self.matrix.shape, 'invalid transformation matrix'
            if abs(LA.det(x) - 1) >= 1e-06:
                print('Warning: det(x) != 1')
            return Lattice( x @ self.matrix )
        # - as scipy.spatial.transformation
        elif isinstance(x, sp.spatial.transform.Rotation):
            return Lattice( x.apply(self.matrix) )
        else:
            print(f"Warning: transformation not understoor: {x}")
        
    # - metrics
    @property
    def matrix(self):
        return self._matrix
    
    @matrix.setter
    def matrix(self, x:np.ndarray):
        self._matrix = x # .round(tbx._PREC)
        self._G = None
        self._reciprocal = None
    
    @property
    def M(self):
        """ alias """
        return self.matrix

    @property
    def metric_tensor(self):
        """ alias """
        return self.G
    
    @property
    def G(self):
        """ metric tensor """
        if self._G is None:
            self._G = (self.M @ self.M.T) # .round(tbx._PREC)
        return self._G

    @property
    def reciprocal(self):
        if self._reciprocal is None:
            # self._reciprocal = Lattice.from_metric(LA.inv(self.G))
            V = self.V
            x1, x2, x3 = self.M
            b1 = np.cross(x2, x3) / V
            b2 = np.cross(x1, x3) / V
            b3 = np.cross(x1, x2) / V
            self._reciprocal = Lattice.from_matrix(np.array((b1, b2, b3)))
        return self._reciprocal
    
    # - properties
    @property
    def volume(self):
        return np.sqrt(LA.det(self.G))
    
    @property
    def V(self):
        """ alias """
        return self.volume
    
    @property
    def a(self):
        return self.length((1,0,0)) # np.sqrt( self.e1 @ self.e1 )

    @property
    def b(self):
        return self.length((0,1,0)) # np.sqrt( self.e2 @ self.e2 )

    @property
    def c(self):
        return self.length((0,0,1)) # np.sqrt( self.e3 @ self.e3 )

    @property
    def al(self):
        return self.angle((0,1,0),
                          (0,0,1), degrees=True)

    @property
    def be(self):
        return self.angle((1,0,0),
                          (0,0,1), degrees=True)
    
    @property
    def ga(self):
        return self.angle((1,0,0),
                          (0,1,0), degrees=True)

    @property
    def abc(self):
        return np.array((self.a, self.b, self.c), dtype=float)

    @property
    def angles(self):
        return np.array((self.al, self.be, self.ga), dtype=float)

    @property
    def scalar(self) -> np.ndarray:
        return np.concatenate((self.abc, self.angles))

    # - functions 
    def angle_between(self, x1: np.ndarray, x2: np.ndarray, x3:np.ndarray, degrees=False):
        """ vertex at x2. default vertex == origin """
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        x3 = np.asarray(x3)
        x12 = x1 - x2
        x32 = x3 - x2
        rv = np.arccos(x12 @ self.G @ x32 / (self.length(x12) * self.length(x32)))
        if degrees:
            return 180 / np.pi * rv
        return rv

    def distance_between(self, x1:np.ndarray, x2: np.ndarray, degrees=False):
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        x12 = x2 - x1
        return self.length(x12)
    
    def angle(self, v1:np.ndarray, v2:np.ndarray, degrees=False):
        r""" :math:`cos \theta := v_1 \cdot v_2 / |v_1||v_2|` """
        return self.angle_between(v1, np.array((0,0,0)), v2, degrees)
        
    def length(self, v1:np.ndarray):
        r""" :math:`v_1 \cdot v_1 == v_1^T G v_1` """
        return np.sqrt(v1 @ self.G @ v1)

    def dhkl(self, h, k, l):
        return 1 / self.reciprocal.length((h, k, l))

    # FIXME under construction 
    @property
    def laue(self):
        """ return crystal system """
        ... 
        return 'not implemented'

    # Lattice
  
    