# -*- coding: utf-8 -*-
r"""


.. math::

    C_{hkl} = \sum_{i,k}^{3} \sum_{j,l}^{2} G_{ijkl} E_{ijkl}

Created on Fri Jun  3 10:06:56 2022

@author: pmetz1
"""
# built-ins
import functools

# 3rd party
import numpy as np
from numpy import linalg as LA

# package imports
# from .lattice import Lattice # Lattice
from .elastic import Stroh   # anisotropic strain
from .geometric import Dislocation # crystal reference frame
from . import toolbox as tbx # orthogonal, unit_vectors, float_tol


_SMALL = 1e-12


class MLS():
    r"""
    Implements the geometric and elastic portions of the MLS contrast factor
    calculation

    .. math::

        C_{hkl} = \sum_{i,m} \sum_{j,n} G_{ijmn} E_{ijmn}

    Reference
    ---------
    Martinez-Garcia, Leoni, & Scardi (2009) "A general approach for
        determining the diffraction contrast factor of straight-line
        dislocations." Acta Cryst. A65, 109–119. `doi:10.1107/S010876730804186X <https://dx.doi.org/10.1107/S010876730804186X>`_
    """
    def __repr__(self):
        return f'<Martinez(\ndislocation={self.dislocation.__repr__()},\ncij={self.stroh.cij} @ {hex(id(self))}>'

    def __init__(self,
                 dislocation: Dislocation=None, # dislocation geometry (carries around a lattice instance)
                 cij: np.ndarray=None,          # elastic tensor -> anisotropic strain
                 ) -> None:
        """ constructu from CIJ representation of elastic stiffness (6,6) """
        self.dislocation = dislocation             # Dislocation instance
        self.stroh = Stroh(cij, self.dislocation.laue) # Stroh instance

    def Gijmn(self, s:np.ndarray) -> np.ndarray:
        r"""
        :math:`G_{ijmn} = \tau_i \tau_j \tau_m \tau_n`.

        See `pymls.contrast.Dislocation.Gijmn`.

        Parameters
        ----------
        np.ndarray (3,)
            Diffraction vector.


        Returns
        -------
        Gijmn :  np.ndarray (i, j, m, n) == (3, 2, 3, 2) real float
            Product of direction cosines :math:`\tau_i`.
        """
        return self.dislocation.Gijmn(s)

    # - indexed on alpha only
    @functools.cached_property
    def D(self) -> np.ndarray:
        r"""       
        .. math::

            D_{\alpha} = - \frac{(L_{\alpha} \cdot b_{v})}{|b_{v}|(A_{\alpha} \cdot L_{\alpha})}

        where :math:`A_{\alpha}` and :math:`L_{\alpha}` are equal to the `a`
        and `b` Stroh eigenvectors.

        NB this differs from Stroh's D_alpha coefficients by :math:`1/|\vec{b}|`


        Returns
        -------
        np.ndarray (a,) -> (3,) complex (length**-1)
            Component of :math:`E_{ijmn}` calculation.


        Reference
        ---------
        c.f. eqn. 13, `Martinez-Garcia, Leoni, Scardi (2009). <https://dx.doi.org/10.1107/S010876730804186X>`_
        """
        b = self.dislocation.uvw # b_j
        modb = self.dislocation.length(self.dislocation.uvw) # |b_j|
        A = self.stroh.A # column vectors
        L = self.stroh.L # column vectors
        D = np.empty((3,), dtype=complex)
        for i in range(3):
            D[i] = -(L[:,i] @ b) / (modb * (A[:,i] @ L[:,i]))
        return D

    @functools.cached_property
    def x(self) -> np.ndarray:
        r"""
        :math:`x = \arctan(\Re{(p)}/\Im{(p)})`


        Returns
        -------
        np.ndarray (a,) -> (3,) (radians)
            Component of :math:`E_{ijmn}` calculation.


        Reference
        ---------
        c.f. eqn. 26, `Martinez-Garcia, Leoni, Scardi (2009). <https://dx.doi.org/10.1107/S010876730804186X>`_
        """
        return np.arctan(self.stroh.P.real / self.stroh.P.imag) # real

    @functools.cached_property
    def y(self) -> np.ndarray:
        r"""
        
        .. math::
            
            y = \arctan(\frac{(\Re{(p_i)} - \Re{(p_j)})}{(\Im{(p_i)} + \Im{(p_j)})})


        Returns
        -------
        np.ndarray (a,a`) -> (3,3) radians, real, skew-symmetric
            Component of :math:`E_{ijmn}` calculation.


        Reference
        ---------
        c.f. eqn. 26, `Martinez-Garcia, Leoni, Scardi (2009). <https://dx.doi.org/10.1107/S010876730804186X>`_
        """
        p3 = self.stroh.P * np.ones((3,3))
        return np.arctan( (p3.T - p3).real / (p3.T + p3).imag ) # real

    @functools.cached_property
    def z(self) -> np.ndarray:
        r"""
        :math:`z = \arctan(\Gamma_1(\alpha) / \Gamma_2(\alpha))`


        Returns
        -------
        np.ndarray (a,a`) -> (3,3) radians, real
            Component of :math:`E_{ijmn}` calculation.


        Reference
        ---------
        c.f. eqn. 26, `Martinez-Garcia, Leoni, Scardi (2009). <https://dx.doi.org/10.1107/S010876730804186X>`_
        """
        return np.arctan(self.gamma1 / self.gamma2 ) # real

    @functools.cached_property
    def F(self) -> np.ndarray:
        r"""
        .. math::

            F(p_{\alpha}) = ( \Re[p_{\alpha}] - \Re[p_{\alpha}^`] )^2 + ( \Im[p_{\alpha}] - \Im[p_{\alpha}^`] )^2


        Returns
        -------
        np.ndarray (a, a`) -> (3, 3) real, symmetric
            Component of :math:`E_{ijmn}` calculation.


        Reference
        ---------
        c.f. eqn. 26, `Martinez-Garcia, Leoni, Scardi (2009). <https://dx.doi.org/10.1107/S010876730804186X>`_
        """
        p3 = self.stroh.P * np.ones((3,3))
        A = p3.T - p3
        return A.real ** 2 + A.imag ** 2 # real

    @functools.cached_property
    def Q(self) -> np.ndarray:
        r"""
        .. math::

            Q(p_{\alpha}) = (|p_{\alpha}|^2 - |p_{\alpha^`}|^2 )^2 + 4 (Re[p_{\alpha}] - Re[p_{\alpha^`}]) (|p_{\alpha^`}|^2 Re[p_{\alpha}] - |p_{\alpha}|^2 Re[p_{\alpha^`}])


        Returns
        -------
        np.ndarray (a, a`) == (3, 3) real, symmetric
            Component of :math:`E_{ijmn}` calculation.


        Reference
        ---------
        c.f. eqn. 26, `Martinez-Garcia, Leoni, Scardi (2009). <https://dx.doi.org/10.1107/S010876730804186X>`_
        """
        p     = self.stroh.P
        mod2  = np.abs(self.stroh.P)**2 # (p * np.conjugate(p).T).real
        p3    = p * np.ones((3,3))
        mod23 = mod2 * np.ones((3,3))
        q1 = (mod23.T - mod23)**2
        q2 = 4 * (p3.T - p3).real
        q3 = mod23 * p3.T.real - mod23.T * p3.real
        return q1 + q2 * q3 # real

    # - indexed on alpha & ij(mn)
    # must reduce such that Arg(z) -> scalar and z = complex
    @functools.cached_property
    def delta(self) -> np.ndarray:
        r"""
        :math:`\Delta_{\alpha}^{mn} = arg(A_{m \alpha} D_{\alpha} p_{\alpha}^{(n-1)})`

        for

        :math:`\alpha \in 1,2,3, m \in 1,2,3, n \in 1,2`

        NB :math:`(n-1) \in (1,2)` is an *exponent* (rather than an index) 
        resulting from evaluation of the partial differentials 
        
        .. math::

            \frac{\partial}{\partial x_n} ln(x_1 + p_{\alpha} x_2)



        Returns
        -------
        np.ndarray (a, m, n) == (3, 3, 2) radians, real
            Component of :math:`E_{ijmn}` calculation.


        Reference
        ---------
        c.f. eqn. 26, `Martinez-Garcia, Leoni, Scardi (2009). <https://dx.doi.org/10.1107/S010876730804186X>`_
        """
        rv = np.zeros((3,3,2), dtype=complex)
        I  = np.indices(rv.shape).T # len, 3, 3, 2 - > 2, 3, 3, len
        I  = I.reshape((-1, len(rv.shape))) # -> N x (a,i,j)
        A  = self.stroh.A # A_ia
        D  = self.D # D_a
        P  = self.stroh.P # P_a
        for index in I:
            a, m, n = index
            rv[tuple(index)] = A[m, a] * D[a] * P[a]**int(n+1-1)  # NB A are column eigenvectors
        # return np.arctan(k.imag / k.real) # https://en.wikipedia.org/wiki/Argument
        return np.angle(rv)

    @functools.cached_property
    def gamma1(self) -> np.ndarray:
        r"""
        .. math::

            \Gamma_1( \alpha ) = Im[p_{\alpha}] Im[p_{\alpha^`}] [tan(x_{\alpha}) + tan(x_{\alpha^`})]


        Returns
        -------
        np.ndarray (a, a`) == (3, 3) radians, real, symmetric
            Component of :math:`E_{ijmn}` calculation.


        Reference
        ---------
        c.f. eqn. 27, `Martinez-Garcia, Leoni, Scardi (2009). <https://dx.doi.org/10.1107/S010876730804186X>`_
        """
        p3 = self.stroh.P * np.ones((3,3)) # complex
        x3 = self.x * np.ones((3,3)) # real
        return p3.imag.T * p3.imag * ( np.tan(x3.T) + np.tan(x3) ) # real

    @functools.cached_property
    def gamma2(self) -> np.ndarray:
        r"""
        .. math::

            \Gamma_2( \alpha ) = |p_{\alpha}|^2 - Re[p_{\alpha}] Re[_{\alpha^`}] + Im[p_{\alpha}] Im[p_{\alpha^`}]


        Returns
        -------
        np.ndarray (a, a`) == (3, 3) radians
            Component of :math:`E_{ijmn}` calculation.


        Reference
        ---------
        c.f. eqn. 27, `Martinez-Garcia, Leoni, Scardi (2009). <https://dx.doi.org/10.1107/S010876730804186X>`_
        """
        p3 = self.stroh.P * np.ones((3,3))
        mod23 = np.abs(self.stroh.P)**2 * np.ones((3,3))
        return mod23.T - (p3.real.T * p3.real) + (p3.imag.T * p3.imag) # real

    @functools.cached_property
    def psi(self) -> np.ndarray:
        r"""
        .. math::

              \Psi_{\alpha}^{\alpha'} =  & \frac{ \left|p_{\alpha}\right| }{2 \Im{\left[p_{\alpha}\right]^{2}} }\;                                                   & if\ \alpha = \alpha'   \\
                                         & \frac{ \left|p_{\alpha}\right| }{ \Im{\left[p_{\alpha}\right]} } \left[\frac{F(p_{\alpha})}{Q(p_{\alpha})}\right]^{1/2}\; & if\ \alpha \ne \alpha' \\


        Returns
        -------
        np.ndarray (a, a`) == (3, 3) real
            Component of :math:`E_{ijmn}` calculation.


        Reference
        ---------
        c.f. eqn. 18.1, `Martinez-Garcia, Leoni, Scardi (2009). <https://dx.doi.org/10.1107/S010876730804186X>`_
        """
        rv = np.zeros((3,3))
        m = ~np.eye(3, dtype=bool) # a != a`
        p = self.stroh.P # p_a
        modp = np.abs(p) # |p_a|
        pre = (modp / p.imag * np.ones((3,3))).T
        rv[m] = pre[m] * (self.F[m] / self.Q[m]) ** 0.5  # a != a`
        rv[~m] = modp / (2 * p.imag**2) # a == a`
        return rv

    @functools.cached_property
    def phi(self) -> np.ndarray:
        r"""
        .. math::

            \Phi_{ij\alpha}^{mn\alpha^`} = 2 |A_{i\alpha}||A_{i\alpha^`}||D_{\alpha}||D_{\alpha^`}||P_{\alpha}^{(j-1)}||P_{\alpha^`}^{(n-1)}|

        NB :math:`(n-1) \in (1,2)` is an *exponent* (rather than an index) 
        resulting from evaluation of the partial differentials 
        
        .. math::

            \frac{\partial}{\partial x_n} ln(x_1 + p_{\alpha} x_2)


        Returns
        -------
        np.ndarray (i,j,a,m,n,a`) == (3,2,3,3,2,3) real valued.
            Component of :math:`E_{ijmn}` calculation.


        Reference
        ---------
        c.f. eqn. 18.2, `Martinez-Garcia, Leoni, Scardi (2009). <https://dx.doi.org/10.1107/S010876730804186X>`_
        """
        # - alias
        modA = np.abs(self.stroh.A) # |A_ia| == | stroh eigen vectors |
        modD = np.abs(self.D) # |D_a|  == | quantity containing direction cosines |
        modP = np.abs(self.stroh.P) # |P_a|  == | stroh eigen values |
        # - setup
        rv = np.zeros((3,2,3,3,2,3), dtype=float) # <--- note this is a real valued tensor
        I = np.indices(rv.shape).T
        I = I.reshape((-1, len(rv.shape)))
        for index in I:
            i, j, a, m, n, b = index
            rv[tuple(index)] = 2 * modA[i, a] * modA[m, b] * modD[a] * modD[b] * modP[a]**(j+1-1) * modP[b]**(n+1-1)
        return rv

    # FIXME NB amn | bij but -> ija mnb (revisit indexing throughout)
    @functools.cached_property
    def Eijmn(self) -> np.ndarray:
        r"""
        .. math::

            E_{ijmn} = \Sigma_{\alpha, \alpha'}^{3} \Psi_{\alpha}^{\alpha'} \Phi_{ij\alpha}^{mn\alpha'} \left[ \cos(\Delta_{\alpha}^{mn} + x_{\alpha}) \cos(\Delta_{ij}^{\alpha'} - y_{\alpha}^{\alpha'}) + \sin(\Delta_{\alpha}^{mn}) \sin(\Delta_{ij}^{\alpha'} - z_{\alpha}^{\alpha'}) \right]

        Returns
        -------
        np.ndarray (i,j,m,n) == (3,2,3,2) real valued and positive
            Elastic component of the dislocation contrast factor.


        Reference
        ---------
        c.f. eqn. 17, `Martinez-Garcia, Leoni, Scardi (2009). <https://dx.doi.org/10.1107/S010876730804186X>`_
            """
        # =============================================================================
        #         # - setup return array
        #         rv = np.zeros((3,2,3,3,2,3), dtype=float) # <--- note this is a real valued tensor
        #         I = np.indices(rv.shape, dtype=np.intp).T # len(shape), *shape -> *shape. len(shape)
        #         I = I.reshape((-1, len(rv.shape)))
        #         # - alias objects
        #         psi = self.psi  # (a, a`)  == (3,3)
        #         phi = self.phi  # (i,j,a,m,n,a`) == (3,2,3,3,2,3)
        #         delta = self.delta  # (a, m, n) == (3,3,2)
        #         x = self.x  # (a,) == (3,)
        #         y = self.y  # (a, a`) == (3,3)
        #         z = self.z  # (a, a`) == (3,3)
        #         # - compute elements
        #         for index in I:
        #             i, j, a, m, n, b = index
        #             rv[tuple(index)] = \
        #                 psi[a,b] * phi[i,j,a,m,n,b] * (\
        #                     np.cos(delta[a,m,n] + x[a]) * np.cos(delta[b,i,j] - y[a,b]) \
        #                   + np.sin(delta[a,m,n]) * np.sin(delta[b,i,j] + z[a,b])
        #                     )
        #         # - contract
        #         # equivalent to rv.sum(axis=2).sum(axis=-1)
        #         rv = np.einsum('ijamnb->ijmn', rv) # .round(tbx._PREC)
        #         return rv
        # =============================================================================
        # - alternative
        # =============================================================================
        #         pp = np.einsum('ab,ijamnb->ijamnb', psi, phi)
        #         a1 = np.transpose(delta.T - np.eye(3)*x)
        #         a2 = np.transpose(delta.T - y)
        #         b1 = delta
        #         b2 = np.transpose(delta.T - z)
        #         cos = np.einsum('amn,bij->ijamnb', np.cos(a1), np.cos(a2))
        #         sin = np.einsum('amn,bij->ijamnb', np.sin(b1), np.sin(b2))
        #         con = np.einsum('ijamnb,ijamnb,ijamnb->ijmn', pp, cos, sin)
        #         ...
        # =============================================================================
        # Try again...
        # =============================================================================
        C = np.einsum('amn,bij->ijamnb', self._c1, self._c2) + np.einsum('amn,bij->ijamnb', self._s1, self._s2) # (i,j,a,m,n,b) == (3,2,3,3,2,3)
        B = np.einsum('ijamnb,ijamnb->ijamnb', self.phi, C)
        A = np.einsum('ab,ijamnb->ijmn', self.psi, B)
        return A # (i,j,m,n) == (3,2,3,2)

    @functools.cached_property
    def _c1(self) -> np.ndarray:
        r"""
        .. math::

            \_c_1 = \cos(\Delta_{\alpha}^{mn} + x_{\alpha})

        Returns
        -------
        np.ndarray (3,3,2) real
            Component of :math:`E_{ijmn}` calculation.
        """
        # return np.cos( np.einsum('amn,a->amn', self.delta, self.x) ) # addition not multiplication
        A = np.zeros((3,3,2))
        for n in range(2):
            for m in range(3):
                for a in range(3):
                    A[a,m,n] = np.cos(self.delta[a,m,n] + self.x[a])
        return A

    @functools.cached_property
    def _c2(self) -> np.ndarray:
        r"""
        .. math::

            \_c_2 = \cos(\Delta_{ij}^{\alpha'} - y_{\alpha}^{\alpha'})

        Returns
        -------
        np.ndarray (3,3,2) real
            Component of :math:`E_{ijmn}` calculation.
        """
        # return np.cos( np.einsum('bmn,ab->amn', self.delta, -self.y) )
        A = np.zeros((3,3,2))
        for n in range(2):
            for m in range(3):
                for a in range(3):
                    for b in range(3):
                        A[a,m,n] = self.delta[a,m,n] - self.y[a,b] + A[a,m,n] # contract arg
                    A[a,m,n] = np.cos(A[a,m,n]) # eval cos
        return A

    @functools.cached_property
    def _s1(self) -> np.ndarray:
        r"""
        .. math::
            \_s_1 = \sin(\Delta_{\alpha}^{mn})

        Returns
        -------
        np.ndarray (3,3,2) real
            Component of :math:`E_{ijmn}` calculation.
        """
        return np.sin(self.delta)

    @functools.cached_property
    def _s2(self) -> np.ndarray:
        r"""
        .. math::

            \_s_2 = \sin(\Delta_{ij}^{\alpha'} - z_{\alpha}^{\alpha'})

        Returns
        -------
        np.ndarray (3,3,2) real
            Component of :math:`E_{ijmn}` calculation.
        """
        # return np.sin( np.einsum('bmn,ab->amn', self.delta, self.z) )
        A = np.zeros((3,3,2))
        for n in range(2):
            for m in range(3):
                for a in range(3):
                    for b in range(3):
                        A[a,m,n] = self.delta[a,m,n] - self.z[a,b] + A[a,m,n] # contract arg
                    A[a,m,n] = np.sin(A[a,m,n]) # eval sin
        return A

    def Chkl(self, s:np.ndarray) -> float:
        r"""
        .. math::

            C_{hkl} = \sum_{i,m}^{3} \sum_{j,n}^{2} G_{ijmn} E_{ijmn}

        Parameters
        ----------
        s : np.ndarray
            Diffraction vector (3,).


        Returns
        -------
        float
            Dislocation contrast factor for the diffraction vector `hkl`.


        Reference
        ---------
        c.f. eqn. 9, `Martinez-Garcia, Leoni, Scardi (2009). <https://dx.doi.org/10.1107/S010876730804186X>`_
        """
        return np.einsum('ijmn,ijmn->', self.Gijmn(s) ,self.Eijmn)

    # End Martinez

