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
    """ Simple representation of a lattice """
    
    # - state variables
    _G = None
    _reciprocal = None
    
    def __repr__(self):
        return f'<Lattice(a={self.a:.5f}, b={self.b:.5f}, c={self.c:.5f}, alpha={self.al:.5f}, beta={self.be:.5f}, gamma={self.ga:.5f} @ {hex(id(self))} >'
                     
    def __hash__(self):
        return hash((self.a, self.b, self.c, self.al, self.be, self.ga))

    def __eq__(self, other):
        if isinstance(other, Lattice):
            return self.__hash__() == other.__hash__()  # FIXME better to use float tolerance
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __init__(self, matrix:np.ndarray=None) -> None:
        """
        Simple representation of a Lattice.

        Parameters
        ----------
        matrix : np.ndarray, optional
            Vector basis. The default is I(3).

        Returns
        -------
        None
        """
        self.matrix = matrix
        
    # - constructors
    @classmethod
    def from_scalar(cls, x:tuple) -> Lattice:
        """
        Constructor method for Lattice instance.

        Parameters
        ----------
        x : tuple
            Scalar lattice (a, b, c, alpha, beta, gamma).

        Returns
        -------
        Lattice
            Instance of Lattice.

        Reference
        ---------
        Julian (2014) Foundations of Crystallography, p.17 eqn. 1.6    
        """
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
        """
        Constructor method for Lattice instance

        Parameters
        ----------
        x : np.ndarray
            Vector basis (M).

        Returns
        -------
        Lattice
            Instance of Lattice.
        """
        return cls(x)

    # FIXME there must be a linear algebra route to this.
    @classmethod
    def from_metric(cls, x:np.ndarray) -> Lattice:
        r"""
        Constructor method for Lattice instance.
                
        .. math::
            
            \vec{a}\cdot\vec{b} = a\ b\ cos(\theta)
            
            \theta = cos^{-1}\left(\frac{\vec{a}\cdot\vec{b}}{a b}\right)
            

        Parameters
        ----------
        x : np.ndarray
            Lattice metric tensor (G).

        Returns
        -------
        Lattice
            Instance of Lattice.
        """
        abc = np.sqrt(np.diag(x))
        angles = np.array((
            np.arccos(x[1,2] / np.prod(abc[[1,2]])), # arccos(bc cos(alpha) / bc)
            np.arccos(x[0,2] / np.prod(abc[[0,2]])), # arccos(ac cos(beta)  / ac)
            np.arccos(x[0,1] / np.prod(abc[[0,1]]))  # arccos(ab cos(gamma) / ab)
            )) * 180 / np.pi
        return cls.from_scalar(np.concatenate((abc, angles)))
    
    @classmethod
    def dispatch_constructor(cls, x) -> Lattice:
        """
        Dispatcher for Lattice constructor.

        Parameters
        ----------
        x : TYPE
            Scalar vector (6,), vector basis (3,3), or metric tensor (3,3).

        Raises
        ------
        Exception
            Fails provided input doesn't match one of the provided constructor arguments.

        Returns
        -------
        Lattice
            Instance of Lattice.
        """
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
        """ returns new instance of self from self.M """
        return Lattice(self.matrix) # deepcopy(self)
    
    def transform(self, x:(np.ndarray,sp.spatial.transform.Rotation)) -> Lattice:
        """
        Will probably deprecate sp.spatial.Rotation in favor of generalized
        Affine approach.
        """
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
    def matrix(self) -> np.ndarray:
        r"""
        Vector basis of lattice, as in
        
        .. math::
            
            matrix = [\vec{x_1}, \vec{x_2}, \vec{x_3}]
            
                   =
            \begin{bmatrix}
                  ( x_{11} & x_{12} & x_{13} )\\
                  ( x_{21} & x_{22} & x_{23} )\\
                  ( x_{31} & x_{32} & x_{33} )\\
            \end{bmatrix}
        """
        return self._matrix
    
    @matrix.setter
    def matrix(self, x:np.ndarray):
        self._matrix = x # .round(tbx._PREC)
        self._G = None
        self._reciprocal = None
    
    @property
    def M(self):
        """ Alias of `self.matrix`. """
        return self.matrix

    @property
    def metric_tensor(self):
        r"""
        Metric tensor of the crystal lattice.
        
        .. math::
            
            G = M \cdot M^T
        """
        return self.G
    
    @property
    def G(self):
        """ Alias of `self.metric_tensor`. """
        if self._G is None:
            self._G = (self.M @ self.M.T) # .round(tbx._PREC)
        return self._G

    @property
    def reciprocal(self):
        r"""
        Lattice instance from :math:`M^{-1}`.
        
        NB private attribute set on instantiation.
        """
        if self._reciprocal is None:
            self._reciprocal = Lattice.from_matrix(LA.inv(self.M).T)
        return self._reciprocal
    
    # - properties
    @property
    def volume(self):
        r"""
        Lattice volume:
            
        .. math::
           
            V = \sqrt{ det\ G }
        """
        return np.sqrt(LA.det(self.G))
    
    @property
    def V(self):
        r""" Alias of `self.volume`. """
        return self.volume
    
    @property
    def a(self):
        r""" Lattice scalar `a`. """
        return self.length((1,0,0)) # np.sqrt( self.e1 @ self.e1 )

    @property
    def b(self):
        r""" Lattice scalar 'b'. """
        return self.length((0,1,0)) # np.sqrt( self.e2 @ self.e2 )

    @property
    def c(self):
        r""" Lattice scalar 'c'. """
        return self.length((0,0,1)) # np.sqrt( self.e3 @ self.e3 )

    @property
    def al(self):
        r""" Lattice scalar :math:`\alpha`. """
        return self.angle((0,1,0),
                          (0,0,1), degrees=True)

    @property
    def be(self):
        r""" Lattice scalar :math:`\beta`. """
        return self.angle((1,0,0),
                          (0,0,1), degrees=True)
    
    @property
    def ga(self):
        r""" Lattice scalar :math:`\gamma`. """
        return self.angle((1,0,0),
                          (0,1,0), degrees=True)

    @property
    def abc(self):
        r""" Lattice vector magnitudes `(a, b, c)`. """
        return np.array((self.a, self.b, self.c), dtype=float)

    @property
    def angles(self):
        r""" Lattice vector angles :math:`(\alpha, \beta, \gamma)`. """
        return np.array((self.al, self.be, self.ga), dtype=float)

    @property
    def scalar(self) -> np.ndarray:
        r""" Lattice scalars :math:`(a, b, c, \alpha, \beta, \gamma)`. """
        return np.concatenate((self.abc, self.angles))
    
    @property
    def is_orthogonal(self) -> bool:
        r"""  """
        return tbx.is_orthogonal(self.M)

    # - functions 
    def angle_between(self, x1: np.ndarray, x2: np.ndarray, x3:np.ndarray, degrees=False):
        r"""
        Angle :math:`\angle(x_1, x_2, x_3)` with vertex at :math:`x_2`.

        Parameters
        ----------
        x1 : np.ndarray
            Cartesian coordinate 1.
        x2 : np.ndarray
            Cartesian coordinate 2.
        x3 : np.ndarray
            Cartesian coordinate 3.
        degrees : Bool, optional
            Return degrees or radians. The default is False (radians).

        Returns
        -------
        Float
            :math:`\angle(x_1,x_2,x_3) = cos^{-1}(x_{12} \cdot G \cdot x_{23} / (|x_{12}|\ |x_{23}|))`.
        """
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        x3 = np.asarray(x3)
        x12 = x1 - x2
        x32 = x3 - x2
        rv = np.arccos(x12 @ self.G @ x32 / (self.length(x12) * self.length(x32)))
        if degrees:
            return 180 / np.pi * rv
        return rv

    def distance_between(self, x1:np.ndarray, x2: np.ndarray):
        """ Alias for `self.length`. """
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        x12 = x2 - x1
        return self.length(x12)
    
    def angle(self, x1:np.ndarray, x2:np.ndarray, degrees=False):
        r"""
        Angle between cartesian coordinates `x1` and `x2` taking vertex as
        origin.

        Parameters
        ----------
        x1 : np.ndarray
            Cartesian coordinates 1.
        x2 : np.ndarray
            Cartesian coordinates 2.
        degrees : Bool, optional
            Return degrees or radians. The default is radians.

        Returns
        -------
        Float
            :math:`\angle(x_1, O, x_2)`.
        """
        return self.angle_between(x1, np.array((0,0,0)), x2, degrees)
        
    def length(self, v1:np.ndarray):
        r"""
        Distance between points `x1` and `x2`.

        Parameters
        ----------
        x1 : np.ndarray
            Cartesian coordinate 1.
        x2 : np.ndarray
            Cartesian coordinate 2.

        Returns
        -------
        Float
            :math:`\sqrt{x_1 \cdot G \cdot x_2}`.
        """
        return np.sqrt(v1 @ self.G @ v1)

    def dhkl(self, h, k, l):
        r"""
        D-spacing for the corresponding Miller indices.

        Parameters
        ----------
        h, k, l : Int
            Miller indices.

        Returns
        -------
        Float
            Reciprocal lattice spacing.
        """
        return 1 / self.reciprocal.length((h, k, l))

    # FIXME under construction 
    @property
    def laue(self):
        r""" Laue group (may be depricated for Lattice class). """
        ... 
        return 'not implemented'

    # Lattice
  
    