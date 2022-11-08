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
    """
    Implement the geometric and elastic portions of the MLS contrast factor
    calculation
    
    .. math::
        
        C_{hkl} = \sum_{i,m} \sum_{j,n} G_{ijmn} E_{ijmn}
        
    Ref:
        Martinez-Garcia, Leoni, & Scardi (2009) "A general approach for 
        determining the diffraction contrast factor of straight-line
        dislocations." Acta Cryst. A65, 109–119. doi:10.1107/S010876730804186X
    """
    def __repr__(self):
        return f'<Martinez(\ndislocation={self.dislocation.__repr__()},\ncij={self.stroh.cij} @ {hex(id(self))}>'
    
    def __init__(self,
                 # lattice: Lattice=None,         # crystal lattice
                 dislocation: Dislocation=None, # dislocation geometry (carries around a lattice instance)
                 cij: np.ndarray=None,          # elastic tensor -> anisotropic strain
                 ) -> None:
        """ constructu from CIJ representation of elastic stiffness (6,6) """
        self.dislocation = dislocation             # Dislocation instance
        # self.lattice = dislocation.lattice         # Lattice instance
        self.stroh = Stroh(cij, self.dislocation.laue) # Stroh instance        
    
    def Gijmn(self, s):
        """ obtain from lattice & diffraction geometry """
        return self.dislocation.Gijmn(s)
    
    # - indexed on alpha only
    @functools.cached_property
    def D(self):
        r"""
        .. math::
            
            D_{\alpha} = - \frac{(L_{\alpha} \cdot b_{v})}{|b_{v}|(A_{\alpha} \cdot L_{\alpha})}
            
        where :math:`A_{\alpha}` and :math:`L_{\alpha}` are equal to the `a`
        and `b` Stroh eigenvectors.
        
        NB this differs from Stroh's D_alpha coefficients by :math:`1/|\vec{b}|`
        
        returns (3,) complex (length**-1)
        """
        bj = self.dislocation.uvw # b_j
        modb = self.dislocation.length(self.dislocation.uvw) # |b_j|
        Lai = self.stroh.L # L_ai
        ALij = (self.stroh.A @ self.stroh.L) # A_ai dot L_aj -> AL_ij
        return  -1/modb * Lai @ bj @ LA.inv(ALij)
        
    @functools.cached_property
    def x(self):
        """
        arctan(Re{p}/Im{p})
        Martinez-Garcia, Leoni, Scardi (2009) eqn. 26
        returns (a,) -> (3,) (radians)
        """        
        return np.arctan(self.stroh.P.real / self.stroh.P.imag) # real
    
    @functools.cached_property
    def y(self):
        """
        arctan(Re{p_i} - Re{p_j} / Im{p_i} + Im{p_j})
        Martinez-Garcia, Leoni, Scardi (2009) eqn. 26
        returns (a,a`) -> (3,3) (radians)
        """
        p3 = self.stroh.P * np.ones((3,3))
        A = (p3 - p3.T).real
        B = (p3 + p3.T).imag
        return np.arctan( A / B ) # real
    
    @functools.cached_property
    def z(self):
        """
        arctan(\Gamma_1(\alpha) / \Gamma_2(\alpha))
        Martinez-Garcia, Leoni, Scardi (2009) eqn. 26
        returns (a,a`) -> (3,3) (radians)
        """
        return np.arctan(self.gamma1 * LA.inv(self.gamma2)) # real
        
    @functools.cached_property
    def F(self):
        r"""
        .. math::
            
            F(p_{\alpha}) = ( Re[p_{\alpha}] - Re[p_{\alpha}^`] )^2 + ( Im[p_{\alpha}] - Im[p_{\alpha}^`] )^2
        
        Martinez-Garcia, Leoni, Scardi (2009) eqn. 26
        
        return (3,3)
        """
        p3 = self.stroh.P * np.ones((3,3))
        A = p3 - p3.T
        return A.real**2 + A.imag**2 # (p3.real - p3.real.T)**2 + (p3.imag - p3.imag.T)**2
    
    @functools.cached_property
    def Q(self):
        r"""
        .. math::
            
            Q(p_{\alpha}) = (|p_{\alpha}|^2 - |p_{\alpha^`}|^2 )^2 + 4 (Re[p_{\alpha}] - Re[p_{\alpha^`}]) (|p_{\alpha^`}|^2 Re[p_{\alpha}] - |p_{\alpha}|^2 Re[p_{\alpha^`}])
        
        Martinez-Garcia, Leoni, Scardi (2009) eqn. 26
        
        return (3,3)
        """
        p     = self.stroh.P
        mod2  = np.abs(self.stroh.P)**2 # (p * np.conjugate(p).T).real 
        p3    = p * np.ones((3,3))
        mod23 = mod2 * np.ones((3,3))
        q1 = (mod2 - mod2.T)**2
        q2 = 4 * (p3 - p3.T).real
        q3 = mod23.T * p3.real - mod23 * p3.real.T
        return q1 + q2 * q3 # real

    # - indexed on alpha & ij(mn)
    # must reduce such that Arg(z) -> scalar and z = complex
    @functools.cached_property
    def delta(self):
        r"""
        .. math::
            
            \Delta_{\alpha}^{mn} = arg(A_{m \alpha} D_{\alpha} p_{\alpha}^{(n-1)})
            
            for
            
            \alpha \in 1,2,3, m \in 1,2,3, n \in 1,2
            
        NB (n-1) is an *exponent* (rather than an index) resulting from
        evaluation of the partial differentials (Martinez-Garcia et al. eqn. 15)
        
        .. math::
            
            \frac{\partial}{\partial x_n} ln(x_1 + p_{\alpha} x_2)
        
        Martinez-Garcia, Leoni, Scardi (2009) eqn. 26
        
        returns (a,m,n) -> (3,3,2) radians
        """
        rv = np.zeros((3,3,2), dtype=complex)
        I  = np.indices(rv.shape).T # len, 3, 3, 2 - > 2, 3, 3, len
        I  = I.reshape((-1, len(rv.shape))) # -> N x (a,i,j)
        A  = self.stroh.A # A_ai
        D  = self.D # D_a
        P  = self.stroh.P # P_a
        for index in I:
            a, i, j = index
            rv[tuple(index)] = A[a, i] * D[a] * P[a]**int(j-1)
        # return np.arctan(k.imag / k.real) # https://en.wikipedia.org/wiki/Argument
        return np.angle(rv)

    @functools.cached_property
    def gamma1(self):
        r"""
        .. math::
            
            \Gamma_1( \alpha ) = Im[p_{\alpha}] Im[p_{\alpha^`}] [tan(x_{\alpha}) + tan(x_{\alpha^`})]
        
        Martinez-Garcia, Leoni, Scardi (2009) eqn. 27
        
        returns (a,a`) -> (3,3) (radians)
        """
        p3 = self.stroh.P * np.ones((3,3)) # complex
        x3 = self.x * np.ones((3,3)) # real
        return p3.imag * p3.imag.T * ( np.tan(x3) + np.tan(x3.T) ) # real
        
    @functools.cached_property
    def gamma2(self):
        r"""
        .. math::
            
            \Gamma_2( \alpha ) = |p_{\alpha}|^2 - Re[p_{\alpha}] Re[_{\alpha^`}] + Im[p_{\alpha}] Im[p_{\alpha^`}]
        
        Martinez-Garcia, Leoni, Scardi (2009) eqn. 27
        
        returns (a,a`) -> (3,3) (radians)
        """
        p3 = self.stroh.P * np.ones((3,3))
        mod2 = np.abs(self.stroh.P)**2 # np.diag(p3 * np.conjugate(p3)).real
        return mod2 - (p3.real * p3.real.T) + (p3.imag * p3.imag.T) # real
    
    # FIXME corrected but unvalidated
    @functools.cached_property
    def psi(self):
        r"""
        .. math::
            
            case a = a`, a \ne a`
            
            ...
            
        returns (a, a`) -> (3,3)
        """
        p = self.stroh.P # p_a
        modp = np.abs(p) # |p_a|
        a = np.eye(3) * modp / (2 * p.imag**2) # a == a`
        b = np.zeros(self.F.shape) # a != a`
        m = ~np.eye(3, dtype=bool)
        b[m] = ( self.F[m] / self.Q[m] ) ** 0.5
        b *= modp / p.imag
        return a + b
        
    # FIXME corrected but unvalidated
    @functools.cached_property
    def phi(self):
        r"""
        
        .. math::
            
            \Phi_{ij\alpha}^{mn\alpha^`} = 2 |A_{i\alpha}||A_{i\alpha^`}||D_{\alpha}||D_{\alpha^`}||P_{\alpha}^{(j-1)}||P_{\alpha^`}^{(n-1)}|

        NB (n-1) is an *exponent* resulting from evaluation of the partial 
        differentials (rather than an index)
        
        .. math::
            
            \frac{\partial}{\partial x_n} ln(x_1 + p_{\alpha} x_2)
            
        returns (i,j,a,m,n,a`) == (3,2,3,3,2,3) 
        """
        # - alias
        # modA = np.sqrt(self.stroh.A * np.conjugate(self.stroh.A).T).real  # |A_ai|
        # modD = np.sqrt(self.D * np.conjugate(self.D)).real                # |D_a|  
        # modP = np.sqrt(self.stroh.P * np.conjugate(self.stroh.P).T).real  # |P_a| 
        modA = np.abs(self.stroh.A) # |A_ia| == | stroh eigen vectors |
        modD = np.abs(self.D) # |D_a|  == | quantity containing direction cosines |
        modP = np.abs(self.stroh.P) # |P_a|  == | stroh eigen values |
        # - setup
        rv = np.zeros((3,2,3,3,2,3), dtype=float) # <--- note this is a real valued tensor
        I = np.indices(rv.shape).T
        I = I.reshape((-1, len(rv.shape)))
        for index in I:
            i, j, a, m, n, b = index
            rv[tuple(index)] = 2 * modA[a, i] * modA[b, m] * modD[a] * modD[b] * modP[a]**(j-1) * modP[b]**(n-1)
        return rv

    # - finally
    # Having trouble forming multiplications over variety of indexes
    # Delta_a^ij (3,2,3) + y (3,3) e.g.
    @functools.cached_property
    def Eijmn(self):
        """
        Should be real valued and positive with shape (3,2,3,2)
        Contracts over alpha, alpha`.
        (i,j,a,m,n,a`) == (3,2,3,3,2,3) 
        to
        (i,j,m,n) == (3,2,3,2)
        """
        # - setup return array
        rv = np.zeros((3,2,3,3,2,3), dtype=float) # <--- note this is a real valued tensor
        I = np.indices(rv.shape, dtype=np.intp).T # len(shape), *shape -> *shape. len(shape)
        I = I.reshape((-1, len(rv.shape)))
        # - alias objects
        psi, phi, delta = self.psi, self.phi, self.delta
        x, y, z = self.x, self.y, self.z
        # - compute elements
        for index in I:
            i, j, a, m, n, b = index 
            rv[tuple(index)] = \
                psi[a,b] * phi[i,j,a,m,n,b] * (\
                    np.cos(delta[a,m,n] + x[a]) * np.cos(delta[b,i,j] - y[a,b]) \
                  + np.sin(delta[a,m,n]) * np.sin(delta[b,i,j] + z[a,b])
                    )
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
        # - contract
        # equivalent to rv.sum(axis=2).sum(axis=-1)
        rv = np.einsum('ijamnb->ijmn', rv) # .round(tbx._PREC)
        return rv
        
    def Chkl(self, s):
        r""" 
        .. math::
            
            \sum_{i,m}^{3} \sum_{j,n}^{2} G_{ijmn} E_{ijmn}
            
        returns (1,) real scalar
        """
        return np.einsum('ijmn->', self.Gijmn(s) * self.Eijmn)
        
    # End Martinez

