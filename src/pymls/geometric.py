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
            A Lattice instance, scalar parameter set (6,), or vector basis (3,3)
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
    def SG(self) -> int:
        """ Space group number. """
        return self._SGno

    @SG.setter
    def SG(self, x) -> None:
        self._SGno = x

    @property
    def hkl(self) -> np.ndarray:
        """ Miller indices of slip plane. """
        return self._hkl

    @hkl.setter
    def hkl(self, x) -> None:
        self._reset()
        self._hkl = x

    @property
    def uvw(self) -> np.ndarray:
        """ Miller indices of slip vector. """
        return self._uvw

    @uvw.setter
    def uvw(self, x) -> None:
        self._reset()
        self._uvw = x

    @property
    def burgers(self) -> np.ndarray:
        """ Alias of  `self.uvw`. """
        return self.uvw

    @property
    def line(self) -> np.ndarray:
        r""" Line vector defined :math:`R_{p2} \cdot \vec{b}`. """
        return self.Rp2 @ self.uvw

    @property
    def phi(self) -> float:
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
    def phi(self, x) -> None:
        self._reset()
        self._phi = x

    @property
    @tbx.orthogonal
    @tbx.unit_vectors
    def Rp2(self) -> np.ndarray:
        """
        Rotation determined by axis :math:`\chi_2` and angle :math:`\phi`.
        
        NB rotation is defined *clockwise*.

        Returns
        -------
        np.ndarray (3,3)
            Rotation matrix transforming Burgers vector `b` into the line vector `l`.


        Reference
        ---------
        Martinez-Garcia, Leoni, & Scardi (2009) "Diffraction contrast factor of
            dislocations." Acta Cryst A65, 109-119. eqns. 5-6.
        """
        return tbx.rotation_from_axis_angle(vector=self.xi2, angle=-self.phi, degree=True) # MLS shows this rotation of b into l in the negative sense of phi

    @property
    @tbx.unit_vector
    def xib(self) -> np.ndarray:
        """
        Normalized burgers vector.

        Returns
        -------
        np.ndarray (3,)
            Dislocation reference frame vector.

        Reference
        ---------
        Martinez-Garcia, Leoni, & Scardi (2009) "Diffraction contrast factor of
        dislocations." Acta Cryst A65, 109-119. eqn. 4
        """
        return 1 / self.length(self.uvw) * self.uvw @ self.M  # (M^T)^-1 == direct space vector basis

    @property
    @tbx.unit_vector
    def xi1(self) -> np.ndarray:
        """
        Line vector `l` cross slip plane vector `hkl`.

        Returns
        -------
        np.ndarray (3,)
            Dislocation reference frame vector.

        Reference
        ---------
        Martinez-Garcia, Leoni, & Scardi (2009) "Diffraction contrast factor of
        dislocations." Acta Cryst A65, 109-119. eqn. 7
        """
        v = np.cross(self.xi2, self.xi3)
        return v # / LA.norm(v) # should be normalized by construction

    @property
    @tbx.unit_vector
    def xi2(self) -> np.ndarray:
        """
        Normalized slip plane normal vector.

        Returns
        -------
        np.ndarray (3,)
            Dislocation reference frame vector.

        Reference
        ---------
        Martinez-Garcia, Leoni, & Scardi (2009) "Diffraction contrast factor of
        dislocations." Acta Cryst A65, 109-119. eqn. 3
        """

        return 1 / self.reciprocal.length(self.hkl) * self.hkl @ self.reciprocal.M

    @property
    @tbx.unit_vector
    def xi3(self) -> np.ndarray:
        """
        Normalized line vector.

        Returns
        -------
        np.ndarray (3,)
            Dislocation reference frame vector.

        Reference
        ---------
        Martinez-Garcia, Leoni, & Scardi (2009) "Diffraction contrast factor of
        dislocations." Acta Cryst A65, 109-119. eqn. 5
        """
        return self.Rp2 @ self.xib

    @property
    @tbx.orthogonal
    @tbx.unit_vectors
    def P(self) -> np.ndarray:
        r"""
        The matrix transforming the orthogonal O{ijk} frame into the 
        dislocation reference frame {e1, e2, e3}.
        
        .. math::

            \left[ e_1 e_2 e_3 \right] = P \left[i j k \right]

            and

            P = \xi_{ij}

        Returns
        -------
        np.ndarray (3,3)

        Reference
        ---------
            NB MLS (2009) type this in row-major format
        """
        if self._P is None:
            self._P = np.array((self.xi1, self.xi2, self.xi3))
        return self._P

    # FIXME
    # equation 8 is somewhat perplexing-- it gives e_i = PM[abc], but if M
    # is the reciprocal lattice matrix, and [abc] is the crystal lattice matrix,
    # this is the transform of the identity matrix.
    @property
    @tbx.orthogonal
    @tbx.unit_vectors
    def e(self) -> np.ndarray:
        """
        The orthonormal reference frame of the dislocation {e1,e2,e3} transformed
        into the crystal reference frame C{a,b,c}.

        Returns
        -------
        np.ndarray (3,3)
            Dislocation reference frame.

        Reference
        ---------
        Martinez-Garcia, Leoni, & Scardi (2009) "Diffraction contrast factor of
        dislocations." Acta Cryst A65, 109-119. eqn. 8
        """
        if self._e is None:
            self._e = self.P @ self.reciprocal.M @ self.M
        return self._e

    @property
    def e1(self) -> np.ndarray:
        """ Dislocation reference frame 1. """
        return self.e[0]

    @property
    def e2(self) -> np.ndarray:
        """ Dislocation reference frame 2. """
        return self.e[1]

    @property
    def e3(self) -> np.ndarray:
        """ Dislocation reference frame 3. """
        return self.e[2]

    def t1(self, s:tuple) -> float:
        r""" Direction cosine between the diffraction vector `s` and :math:`e_1`. """
        # return s / LA.norm(s) @ self.e1
        return s @ self.reciprocal.M / self.reciprocal.length(s) @ self.e1
        
    def t2(self, s:tuple) -> float:
        r""" Direction cosine between the diffraction vector `s` and :math:`e_2`. """
        # return s / LA.norm(s) @ self.e2
        return s @ self.reciprocal.M / self.reciprocal.length(s) @ self.e2
        
    def t3(self, s:tuple) -> float:
        r""" Direction cosine between the diffraction vector `s` and :math:`e_3`. """
        #return s / LA.norm(s) @ self.e3
        return s @ self.reciprocal.M / self.reciprocal.length(s) @ self.e3
        
    def tau(self, s:np.ndarray) -> np.ndarray:
        r"""
        Direction cosines between diffraction vector `s` and slip reference
        system.

        .. math::

            \tau_i = \vec{s^*}/s^* \cdot \vec{e}_i

        Parameters
        ----------
        s : np.ndarray (3,)
            Diffraction vector.

        Returns
        -------
        np.ndarray (3,)
            Vector of direction cosines.
        """
        return np.array((
            self.t1(s),
            self.t2(s),
            self.t3(s)
            ), dtype=float) # .round(tbx._PREC)

    def Gijmn(self, s:np.ndarray) -> np.ndarray:
        r"""
        Geometric component of the dislocation contrast factor.

        .. math::

            G_{ijmn} = \tau_i \tau_j \tau_m \tau_n,

            (i,m) \in 1,2,3,

            (j,n) \in 1,2.

        Parameters
        ----------
        s : np.ndarray (3,)
            Diffraction vector `s`.

        Returns
        -------
        rv : np.ndarray (3,2,3,2)
            Geometric component of the dislocation contrast factor.

        Reference
        ---------
        Martinez-Garcia, Leoni, & Scardi (2009) "Diffraction contrast factor of
        dislocations." Acta Cryst A65, 109-119. eqn. 10
        """
        rv = np.zeros((3,2,3,2))
        I = np.indices((3,2,3,2)).T.reshape((-1,4))
        tau = self.tau(s)
        for index in I:
            rv[tuple(index)] = np.prod([tau[i] for i in index])
        return rv

    def visualize(self) -> tuple:
        """
        Simple rendering of unit cell and relevant reference frames.

        Returns
        -------
        (fig, ax) : tuple
            (matplotlib.figure, matplotlib.axis).
        """
        o = np.array((0,0,0))
        # - unit cell
        fig, ax = tbx.plot_cell(self)
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