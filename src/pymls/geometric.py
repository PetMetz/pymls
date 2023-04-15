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
#       Probably better to store the Lattice as an attribute and alias it (i.e. as L) for brevity
class Dislocation(lattice.Lattice):
    """
    Slip reference frame for the slip system {hkl}<uvw> and geometric
    aspects of the dislocation contrast factor.
    
    Reference
    ---------
    Martinez-Garcia, Leoni, & Scardi (2009) "Diffraction contrast factor of
    dislocations." Acta Cryst A65, 109-119.
    """
    # - class state
    _Rp2 = None
    _e   = None
    _P   = None

    def _reset(self):
        """ Reset class state. """
        self._Rp2 = None
        self._e   = None
        self._P   = None

    def __repr__(self):
        return f'<Dislocation(hkl={self.hkl}, uvw={self.uvw}, phi={self.phi}) @ {hex(id(self))}>'

    # - instance
    def __init__(self, lattice: lattice.Lattice,
                       hkl: np.ndarray,
                       uvw: np.ndarray,
                       phi: float,
                       SGno: int=None
                       ) -> None:
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
        """
        # FIXME this needs to __init__ to use as a superclass, so a dispatch doesn't work here
        super().__init__(super().dispatch_constructor(lattice).matrix) 
        self.hkl = hkl
        self.uvw = uvw
        self.phi = phi
        self.SG = SGno
            
    @property
    def SG(self):
        """ Space group number. """
        return self._SGno
    
    @SG.setter
    def SG(self, x):
        self._SGno = x
        
    @property
    def hkl(self):
        """ Miller indices of slip plane. """
        return self._hkl
    
    @hkl.setter
    def hkl(self, x):
        self._reset()
        self._hkl = x

    @property
    def uvw(self):
        """ Miller indices of slip vector. """
        return self._uvw
    
    @uvw.setter
    def uvw(self, x):
        self._reset()
        self._uvw = x
    
    @property
    def burgers(self):
        """ Alias of  `self.uvw`. """
        return self.uvw
    
    @property
    def line(self):
        r""" Line vector defined :math:`R_{p2} \cdot \vec{b}`. """
        return self.Rp2 @ self.uvw
    
    @property
    def phi(self):
        """
        Angle between dislocation line and Burgers vector describes
        dislocation character, :math:`\phi`. [1]
        
        "The vector l is obtained by rotating the Burgers vector b clockwise by
        an angle phi (dislocation character) around e2." [2]

        Reference
        ---------
        [1] Wilkens, M. (1970). The determination of density and distribution of
                dislocations in deformed single crystals from broadened X‐ray
                diffraction profiles. Physica Status Solidi (A), 2(2), 359–370.
                https://doi.org/10.1002/pssa.19700020224
        [2] Martinez-Garcia, Leoni, and Scardi (2009). Diffraction contrast
                factor of dislocations. Acta Cryst. A65, 109–119.
        """
        return self._phi
    
    @phi.setter
    def phi(self, x):
        self._reset()
        self._phi = x
       
    @property
    @tbx.orthogonal
    @tbx.unit_vectors
    def Rp2(self):
        """ 
        ...
        
        Reference
        ---------
        ...
        """
        return tbx.rotation_from_axis_angle(vector=self.xi2, angle=self.phi, degree=True) # MLS shows this rotation of b into l in the negative sense of phi
    
    @property
    @tbx.unit_vector
    def xi2(self):
        """
        ...
        
        Reference
        ---------
        MLS (2009) eqn. 3
        """
        return self.reciprocal.M @ self.hkl / self.reciprocal.length(self.hkl)
    
    @property
    @tbx.unit_vector
    def xib(self):
        """
        Normalized burgers vector
        
        Reference
        ---------
        MLS (2009) eqn. 4
        """
        return self.M @ self.uvw / self.length(self.uvw)
    
    @property
    @tbx.unit_vector
    def xi3(self):
        """
        ...
        
        Reference
        ---------
        MLS (2009) eqn. 5
        """
        return self.Rp2 @ self.xib
    
    @property
    @tbx.unit_vector
    def xi1(self):
        """
        ...
        
        Reference
        ---------
        MLS (2009) eqn. 7
        """
        return np.cross(self.xi2, self.xi3)
        
    @property
    @tbx.orthogonal
    @tbx.unit_vectors
    def P(self):
        r""" 
        .. math::
            
            \left[ e_1 e_2 e_3 \right] = P \left[i j k \right]
            
            and
            
            P = \xi_{ij}
            
            NB MLS (2009) type this in row-major format
        
        """
        if self._P is None:
            self._P = np.array((self.xi1, self.xi2, self.xi3))
        return self._P
    
    
    # equation 8 is somewhat perplexing-- it gives e_i = PM[abc], but if M
    # is the reciprocal lattice matrix, and [abc] is the crystal lattice matrix,
    # this is the transform of the identity matrix.
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
            self._e = self.P @ self.reciprocal.M @ self.M
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
        # version 1
        # return np.sqrt(1 - self.t2(s)**2 - self.t3(s)**2)
        # version 2
        v1 = s / LA.norm(s) # should just be a unit vector/ self.reciprocal.length(s)
        v2 = self.e1 # should be unit vector by construction / self.length(self.e1)
        return v1 @ v2
    
    # @functools.lru_cache(maxsize=100)
    def t2(self, s:tuple) -> float:
        r"""
        direction cosines
        
        .. math::
            
            \tau_i = \frac{\vec{d}^*}{d^*} \cdot \vec{e}_i
        
        where math:`\vec{d}^*` is coincident with the diffraction vector and 
        :math:`\vec{e}_i` is a vector in the dislocation reference frame
        
        returns (1,)
        """
        # return self.reciprocal.angle(s, self.hkl)
        # Gstar = self.reciprocal.G
        # version 1
        # n = s @ Gstar @ self.hkl  # vector @ transform @ vector -> scalar
        # d = (self.hkl @ Gstar @ self.hkl)**0.5 * (s @ Gstar @ s)**0.5 # length * length
        # return n / d
        # version 2
        # v1 = s / self.reciprocal.length(s)
        # v2 = self.hkl / self.reciprocal.length(self.hkl)
        # return v1 @ Gstar @ v2
        # version 3
        v1 = s / LA.norm(s) # should just be a unit vector self.reciprocal.length(s)
        v2 = self.e2  # should be unit vector by construction  / self.length(self.e2)
        return v1 @ v2
    
    # @functools.lru_cache(maxsize=100)
    def t3(self, s:tuple) -> float:
        """ direction cosines. s == diffracting plane. returns (1,) """
        # M = self.M
        # Gstar = self.reciprocal.G
        # G = self.G # typo in eqn 11?
        # version 1
        # n = s @ M.T @ self.Rp2 @ LA.inv(M.T) @ self.uvw # vector @ transform @ vector -> scalar
        # d = (self.uvw @ Gstar @ self.uvw )**0.5 * (s @ Gstar @ s)**0.5  # length * length
        # return n / d
        # version 2
        # v1 = s / self.reciprocal.length(s)
        # v2 = self.uvw / self.reciprocal.length(self.uvw)
        # T  = self.M.T @ self.Rp2 @ LA.inv(self.M.T)
        # return v1 @ T @ v2
        # version 3
        v1 = s / LA.norm(s) # should just be a unit vector/ self.reciprocal.length(s)
        v2 = self.e3 # should be unit vector by construction  / self.length(self.e3)
        return v1 @ v2

    # @functools.lru_cache(maxsize=100)
    def tau(self, s:tuple) -> np.ndarray:
        r""" direction cosines :math:`\tau_i = \vec{s^*}/s^* \cdot \vec{e}_i` """
        return np.array((
            self.t1(s),
            self.t2(s),
            self.t3(s)
            ), dtype=float) # .round(tbx._PREC)

    def Gijmn(self, s) -> np.ndarray:
        r"""
        .. math::
            
            G_{ijmn} = \tau_i \tau_j \tau_m \tau_n,
            (i,m) = 1,2,3,
            (j,n) = 1,2.
        """
        a = np.zeros((3,2,3,2)) # .reshape((-1,4))
        I = np.indices((3,2,3,2)).T.reshape((-1,4))
        tau = self.tau(s) # (-1,)
        for index in I:
            a[tuple(index)] = np.product([tau[i] for i in index])
        return a
        # rv = np.product(tau[I], axis=1) # NB this works because tau is 1D, hence I is treated as an integer mask
        # return rv.reshape((3,2,3,2)) # .round(tbx._PREC)
    
    def visualize(self):
        from .toolbox import plot_cell
        o = np.array((0,0,0))
        # - unit cell        
        fig, ax = plot_cell(self)
        # - lattice vectors
        for v, s in zip(self.matrix, ('a', 'b', 'c')):
            ax.plot(*np.transpose((o, v))) # (xs, ys, zs, *args)
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