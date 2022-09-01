# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:39:43 2022

@author: pmetz1
"""

# 3rd party
import numpy as np
from numpy import linalg as LA

# package
from . import lattice
from . import toolbox as tbx


# FIXME I'm not sure subclassing a Lattice is the right approach, but it makes the included expressions more compact.
#       Probably better to store the Lattice as an attribute and alias it (i.e. as L) for compactness
class Dislocation(lattice.Lattice):
    """
    Martinez-Garcia, Leoni, & Scardi (2009) "Diffraction contrast factor of
    dislocations." Acta Cryst A65, 109-119.
    
    construct slip reference frame for the slip system {hkl}<uvw> for the
    crystal system defined by.
    
    """
    # - class state
    _Rp2 = None
    _e   = None
    _P   = None

    def _reset(self):
        self._Rp2 = None
        self._e   = None
        self._P   = None

    def __repr__(self):
        return f'<Dislocation(hkl={self.hkl}, uvw={self.uvw}, phi={self.phi}) @ {hex(id(self))}>'

    # - instance
    def __init__(self, lattice: (lattice.Lattice, np.ndarray, tuple),
                       hkl: np.ndarray,
                       uvw:np.ndarray,
                       phi:float,
                       SGno:int=None
                       ):
        r"""

        Parameters
        ----------
        L : lattice.Lattice
            A Lattice instance, scalar parameter set, or matrix
        hkl : np.ndarray
            (hkl) describing :math:`\vec{n}`, the normal to the slip plane.
        uvw : np.ndarray
            [uvw] describing :math:`\vec{b}`, the burgers vector.
        phi : float
            "Dislocation character." 0 <= phi <= 90 for screw -> mixed -> edge.
        SGno : int, optional
            Space group number. No symmetry operations currently defined.
            The default is None.

        Returns
        -------
        None.

        """
        # FIXME this needs to __init__ to use as a superclass, so a dispatch doesn't work here
        super().__init__(super().dispatch_constructor(lattice).matrix) 
        self.hkl = hkl
        self.uvw = uvw
        self.phi = phi
        self.SG = SGno
            
    @property
    def SG(self):
        return self._SGno
    
    @SG.setter
    def SG(self, x):
        self._SGno = x
        
    @property
    def hkl(self):
        return self._hkl
    
    @hkl.setter
    def hkl(self, x):
        self._reset()
        self._hkl = x

    @property
    def uvw(self):
        return self._uvw
    
    @uvw.setter
    def uvw(self, x):
        self._reset()
        self._uvw = x
    
    @property
    def burgers(self):
        """ alias uvw """
        return self.uvw
    
    @property
    def phi(self):
        return self._phi
    
    @phi.setter
    def phi(self, x):
        """
        Angle between dislocation line and Burgers vector describes
        dislocation character, :math:`\phi`. [1]
        
        Reference
        ---------
        [1] Wilkens, M. (1970). The determination of density and distribution of
                dislocations in deformed single crystals from broadened X‐ray
                diffraction profiles. Physica Status Solidi (A), 2(2), 359–370.
                https://doi.org/10.1002/pssa.19700020224
        """
        self._reset()
        self._phi = x
        
    @property
    @tbx.orthogonal
    @tbx.unit_vectors
    def Rp2(self):
        """ """
        if self._Rp2 is None:
            # - (a) e2 := n = Ha* + Kb* + Lc* with coordinates [HKL] in the basis [a*, b*, c*]
            #       e2 = 1/|n| M @ [h,k,l]
            xi2 = self.reciprocal.M @ self.hkl / np.sqrt( self.hkl @ self.reciprocal.G @ self.hkl)
            # - (b1)
            sinp = np.sin(np.radians(self.phi))
            cosp = np.cos(np.radians(self.phi))
            sinp2 = np.sin(np.radians(self.phi/2)) ** 2
            xi21, xi22, xi23 = xi2
            m1 = 2 * sinp2 * np.array((
                (xi21*xi21,  xi21*xi22, xi21*xi23),
                (xi22*xi21,  xi22*xi22, xi22*xi23), 
                (xi23*xi21,  xi23*xi22, xi23*xi23)
                ))
            m2 = np.array((
                (1,   xi23, xi22),
                (xi23,   1, xi21),     
                (xi22, xi21, 1  )
                ))
            m3 = np.array((
                ( cosp,  sinp, -sinp),
                (-sinp,  cosp,  sinp),
                ( sinp, -sinp,  cosp)
                ))
            self._Rp2 = m1 + (m2 * m3)
        return self._Rp2
    
    
    @property
    @tbx.orthogonal
    @tbx.unit_vectors
    def P(self):
        """ """
        if self._P is None:
            # - (a) e2 := n = Ha* + Kb* + Lc* with coordinates [HKL] in the basis [a*, b*, c*]
            #       e2 = 1/|n| M @ [h,k,l]
            xi2 = self.M @ self.hkl / np.sqrt( self.hkl @ self.G @ self.hkl)
            # - (b) e3 := R(phi, e2) @ [b1, b2, b3]
            xib = self.M @ self.uvw / np.sqrt(self.uvw @ self.G @ self.uvw)
            xi3 = self.Rp2 @ xib
            # - (c) e1 := e2 x e3
            xi1 = np.cross(xi2, xi3)
            self._P = np.array((xi1, xi2, xi3))
        return self._P
    
    # FIXME not sure if this is the correct normalization for nonorthogonal crystal systems
    @property
    @tbx.orthogonal
    @tbx.unit_vectors
    def e(self):
        """
        Martinez-Garcia, Leoni, & Scardi (2009) "Diffraction contrast factor of
        dislocations." Acta Cryst A65, 109-119.
        
        construct slip reference frame for the slip system {hkl}<uvw> for the
        crystal system defined by `lattice`.
        """
        if self._e is None:
            self._e = self.P # @ np.eye(3)
        return self._e

    @property
    def e1(self):
        """ dislocation reference frame """
        return self.e[0]
    
    @property
    def e2(self):
        """ dislocation reference frame """
        return self.e[1]
    
    @property
    def e3(self):
        """ dislocation reference frame """
        return self.e[2]

    # @functools.lru_cache(maxsize=100)  # this isn't a substantial gain, and limits arguments to hashable inputs
    def t1(self, s:tuple) -> float:
        """ direction cosines. s == diffracting plane. returns (1,) """
        return np.sqrt(1 - self.t2(s)**2 - self.t3(s)**2)
    
    # @functools.lru_cache(maxsize=100)
    def t2(self, s:tuple) -> float:
        """ direction cosines. s == diffracting plane. returns (1,) """
        # return self.reciprocal.angle(s, self.hkl)
        Gstar = self.reciprocal.G
        n = s @ Gstar @ self.hkl  # vector @ transform @ vector -> scalar
        d = (self.hkl @ Gstar @ self.hkl)**0.5 * (s @ Gstar @ s)**0.5 # length * length
        return n / d
    
    # @functools.lru_cache(maxsize=100)
    def t3(self, s:tuple) -> float:
        """ direction cosines. s == diffracting plane. returns (1,) """
        M = self.M
        G = self.G
        Gstar = self.reciprocal.G
        n = s @ M.T @ self.Rp2 @ LA.inv(M.T) @ self.uvw # vector @ transform @ vector -> scalar
        d = (self.uvw @ G @ self.uvw )**0.5 * (s @ Gstar @ s)**0.5  # length * length
        return n / d

    # @functools.lru_cache(maxsize=100)
    def tau(self, s:tuple) -> np.ndarray:
        r""" direction cosines :math:`\tau_i = \vec{s^*}/s^* \cdot \vec{e}_i` """
        return np.array((
            self.t1(s),
            self.t2(s),
            self.t3(s)
            ), dtype=float).round(tbx._PREC)

    def Gijmn(self, s) -> np.ndarray:
        r"""
        .. math::
            
            G_{ijmn} = \tau_i \tau_j \tau_m \tau_n,
            (i,m) = 1,2,3,
            (j,n) = 1,2.
        """
        # a = np.zeros((3,2,3,2)) # .reshape((-1,4))
        I = np.indices((3,2,3,2)).T.reshape((-1,4))
        tau = self.tau(s) # (-1,)
        # for index in I:
        #     a[tuple(index)] = np.product([tau[i] for i in index])
        # return a
        rv = np.product(tau[I], axis=1) # NB this works because tau is 1D, hence I is treated as an integer mask
        return rv.reshape((3,2,3,2)).round(tbx._PREC)
    
    def visualize(self):
        from .toolbox import plot_cell
        o = np.array((0,0,0))
        # - unit cell        
        fig, ax = plot_cell(self)
        # - lattice vectors
        for v, s in zip(self.matrix, ('a', 'b', 'c')):
            ax.plot(*np.transpose((o, v)))
            ax.text(*v, s)
        # - dislocation reference frame
        for v, s in zip((self.e), ('e1', 'e2', 'e3')):
            ax.plot(*np.transpose((o, v)), color='blue')
            ax.text(*v, s)
        # - slip system
        for v, s in zip((self.uvw, self.hkl, self.e3), ('burgers', 'normal', 'line')):
            ax.plot(*np.transpose((o, v)), color='red')
            ax.text(*v, s)
        return fig, ax
            
    # End Dislocation