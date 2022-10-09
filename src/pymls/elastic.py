# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:40:09 2022

@author: pmetz1
"""
# built-ins
import functools

# 3rd party
import numpy as np
from numpy import linalg as LA

# package
from . import toolbox as tbx


# --- constants
_LAUE = np.array((
    (1,           2,            3,              4,            5,            6,          7,          8,           9,           10,      11     ), # integer Laue group
    ('triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'tetragonal', 'trigonal', 'trigonal', 'hexagonal', 'hexagonal', 'cubic', 'cubic'), # point symmetry
    ('-1',        '2/m',        'mmm',          '4/m',        '4/mmm',      '-3',       '-3m',      '6/m',       '6/mmm',     'm-3',   'm-3m' )  # crystal system
    ), dtype=object)


# FIXME didn't have a reference handy: https://link.springer.com/content/pdf/bbm%3A978-94-007-0350-6%2F1.pdf
# FIXME more convnient slicing if arranged in rows
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
    (66, 66, 66, 66, 66, 99, 99, 99, 99, 99, 44)
    ), dtype=int)


# --- functions
def generate_index(arg:int) -> tuple:
    """ ## -> (#, #) """
    i, j = divmod(abs(arg), 10)
    return tuple((i-1, j-1))


def get_unique(LaueGroup:int) -> tuple:
    rv = _ELASTIC_RESTRICTIONS[:, LaueGroup-1] # NB zero index
    rv = set(rv[(rv!=0) & (rv!=99)])
    return tuple(rv)


def parse_laue_class(arg:(str, int)) -> int:
    """
    Reduce various common inputs to integer representation.
    
    If crystal system is given, assumes the higher symmetry choice.
    """
    if isinstance(arg, int) and (1 <= arg <= 11): # trivial case
        return arg
    arg = str(arg).lower() # sanitize
    if arg in _LAUE[1]:
        return int(max(np.argwhere(arg == _LAUE[1]))) + 1 # by crystal system
    elif arg in _LAUE[2]:
        return int(max(np.argwhere(arg == _LAUE[2]))) + 1 # by point symmetry
    else:
        raise Exception(f'invalid Laue class: {arg}')


# FIXME can we factor out for/if?
def cij_from_group(*cij, group):
    """ 
    Cij ordered i, j according to unique elements.
    """
    # - helpers
    def A(X):
        return 0.5 * (X[(0,0)] - X[(0,1)]) # NB zero index
    # - setup
    laue = parse_laue_class(group)  # which laue group
    unique = get_unique(laue)       # symmetry restricted unique values
    assert len(unique) == len(cij), f'Unable to map {cij} to ' + ' '.join([f'c{ij}' for ij in unique])
    rv = np.zeros((6,6))            # return value
    # - map
    for index, key in zip(_ELASTIC_RESTRICTIONS[:, 0], _ELASTIC_RESTRICTIONS[:,laue-1]):
        if not bool(key): # short circuit zeros
            continue
        elif key == 99:  # this is only safe because it only occurs last
            rv[generate_index(index)] = A(rv)
        else:
            rv[generate_index(index)] = np.sign(key) * cij[int(np.argwhere(unique==abs(key)))]
    # - diagonally symmetric
    return rv + rv.T - np.diag(rv.diagonal())



# --- classes
class Stroh():
    """ Ref: Ting T.C.T. Elastic Anisotropy. """
    # - class state
    _flag_N = 1
    _flag_eig = 1
    
    @classmethod
    def reset(cls):
        cls._flag_N = 1
        cls._flag_eig = 1
    
    # - handy
    @functools.cached_property
    def conI(cls):
        O = np.zeros((3,3))
        I = np.eye(3)
        conI = np.row_stack((
            np.column_stack((O, I)),
            np.column_stack((I, O))
            ))
        return conI
    
    # FIXME these can just be constants
    # - conventions
    @functools.cached_property
    def _voigt(self):
        return np.array(((0,0), (1,1), (2,2), (1,2), (0,2), (0,1)), dtype=int)
    
    @functools.cached_property
    def _mandel(self):
        return np.array([ np.concatenate((np.ones_like(self._voigt)*self._voigt[i], self._voigt), axis=1) for i in range(6) ], dtype=int)
    
    @staticmethod
    def _compact(X:np.ndarray):
        return np.apply_along_axis(lambda x:''.join(map(str, x)), axis=-1, arr=X)
    
    # - overloads
    def __repr__(self):
        return f'<Stroh(cij=\n{self.cij}, crystalSystem={self.crystalsystem}) @ {hex(id(self))}>'
    
    def __init__(self,
                 cij:np.ndarray=None,
                 crystalSystem:str=None
                 ) -> None:
        """
        constructu from CIJ representation of elastic stiffness (6,6)
        NB properties are cached-- create new instance if you want to update cijkl
        """
        self.cij = cij        
        self.crystalsystem = crystalSystem
        
    @property
    def cijkl(self):
        """ elastic stiffness tensor """
        return self._cijkl

    @cijkl.setter
    def cijkl(self, X):
        self._cijkl = X

    @property
    def cij(self):
        return self.apply_mandel(self._cijkl)
    
    @cij.setter
    def cij(self, X):
        if not X is None:
            self._cijkl = self.invert_mandel(X)
        
    # FIXME this doesn't need to be a class method
    @classmethod
    def to_voigt(cls, i, j, hit=0):
        """ 0 index!!!! """
        if i == j:
            return int(i)
        elif i == 1 and j == 2:
            return 3
        elif i == 0 and j == 2:
            return 4
        elif i == 0 and j == 1:
            return 5
        elif hit == 0:
            return cls.to_voigt(j, i, hit=1)
        else:
            raise Exception(f'unable to match {i} {j} in Voigt scheme')

    def apply_voigt(self, X):
        """
        Apply Voigt reduction scheme to X(3,3) to produce X2(6,).
        11 -> 1; 22 -> 2; 33 -> 3; 23 -> 4; 13 -> 5; 12 -> 6
        """
        return np.array([X[tuple(e)] for e in self._voigt])
    
    def invert_voigt(self, X):
        """
        Given a Voigt reduced vector, reconstruct 2nd order tensor.
        """
        a = np.zeros((3,3))
        for idx, pt in enumerate(self._voigt):
            a[tuple(pt)] = X[idx]
            a[tuple(pt[::-1])] = X[idx]
        return a
        
    def apply_mandel(self, X):
        """
        The extension of Voigt reduction to the 4th rank tensor representing
        proportionality of two 2nd rank tensors.
        """
        return np.reshape([X[tuple(e)] for e in self._mandel.reshape(-1,4)], (6,6))

    # FIXME can't this be vectorized?
    def invert_mandel(self, X):
        """
        Given a Mandel reduced 2nd order tensor, reconstruct 4th order tensor.
        """
        a = np.zeros((3,3,3,3))
        # - form index
        I = self.elastic_symmetry
        I0 = I[0]
        # - apply mapping
        for idx in I0:
            l, r = idx[:2], idx[2:]
            a[tuple(idx)] = X[self.to_voigt(*l), self.to_voigt(*r)]
        # - exhaustive equivalence
        for Ip in I[1:]:
            for idx, jdx in zip(I0, Ip):
                a[tuple(jdx)] = a[tuple(idx)]
        return a
    
    @functools.cached_property
    def elastic_symmetry(self):
        """ form symmetry equivalents ijks = jiks = ksij = ijsk """
        shape = (3,3,3,3)
        I = np.transpose(np.indices(shape)).astype(int) # 3, 3, 3, 3, 4
        I = I.reshape((-1, len(shape))) # -1, 4
        rv = [] 
        # form symmetry equivalents ijks = jiks = ksij = ijsk 
        rv.append( I[:, (0,1,2,3)] )# ijks
        rv.append( I[:, (1,0,2,3)] ) # jiks 
        rv.append( I[:, (2,3,0,1)] ) # ksij
        rv.append( I[:, (0,1,3,2)] ) # ijsk
        return rv

    # FIXME can't this be vectorized?
    def apply_elastic_symmetry(self, X):
        I0, *I = self.elastic_symmetry
        # - exhaustive equivalence
        for Ip in I[1:]:
            for idx, jdx in zip(I0, Ip):
                X[tuple(jdx)] = X[tuple(idx)]
        return X     
    
    @property
    def crystalsystem(self):
        print('Warning: crystal symmetry not yet implemented')
        return self._crystalsystem
    
    @crystalsystem.setter
    def crystalsystem(self, x:str):
        self._crystalsystem = x
    
    # FIXME incomplete
    def apply_crystal_symmetry(self, group):
        """ """
        ...
        
    @functools.cached_property
    def Q(self):
        r"""
        c.f. pp 137 Ting, Elastic Anisotropy.
        
        .. math::
            
            Q_{ik} = C_{i1k1}
            
            Q = 
            \begin{bmatrix}
                C_{11} & C_{16} & C_{15} \\
                C_{16} & C_{66} & C_{56} \\
                C_{15} & C_{56} & C_{55} 
            \end{bmatrix}
        """
        idx = [(0,0), (0,5), (0,4),
               (0,5), (5,5), (4,5),
               (0,4), (4,5), (4,4)] # NB zero index
        return np.reshape([self.cij[e] for e in idx], (3,3))
    
    @functools.cached_property
    def R(self):
        r"""
        c.f. pp 137 Ting, Elastic Anisotropy.
        
        .. math::
            
            R_{ik} = C_{i1k2}
            
            R = 
            \begin{bmatrix}
                C_{16} & C_{12} & C_{14} \\
                C_{66} & C_{26} & C_{46} \\
                C_{56} & C_{25} & C_{45} 
            \end{bmatrix}
        """
        idx = [(0,5), (0,1), (0,3),
               (5,5), (1,5), (3,5),
               (4,5), (1,4), (3,4)] # NB zero index
        return np.reshape([self.cij[e] for e in idx], (3,3))
        
    @functools.cached_property
    def T(self):
        r"""
        c.f. pp 137 Ting, Elastic Anisotropy.
        
        .. math::
            
            T_{ij} = C_{i2k2}
        
            T = 
            \begin{bmatrix}
                C_{66} & C_{26} & C_{46} \\
                C_{26} & C_{22} & C_{24} \\
                C_{46} & C_{24} & C_{44} 
            \end{bmatrix}
        """
        idx = [(5,5), (1,5), (3,5),
               (1,5), (1,1), (1,3),
               (3,5), (1,3), (3,3)] # NB zero index
        return np.reshape([self.cij[e] for e in idx], (3,3))
    
    @functools.cached_property
    def N1(self):
        r"""
        c.f. pp 144 Ting, Elastic Anisotropy.
        
        .. math::
            N_1 = -T^{-1} R^T \ \  [-]
        """
        return -LA.inv(self.T) @ np.transpose(self.R)
    
    @functools.cached_property
    def N2(self):
        r"""
        c.f. pp 144 Ting, Elastic Anisotropy.
        
        .. math::
            N_2 = T^{-1} \ \ [m^2 \cdot N^{-1}]
        """
        return LA.inv(self.T)
    
    @functools.cached_property
    def N3(self):
        r"""
        c.f. pp 144 Ting, Elastic Anisotropy.
        
        .. math::
            N_3 = R T^{-1} R^T - Q \ \ [N \cdot m^{-2}]
        """
        return self.R @ LA.inv(self.T) @ np.transpose(self.R) - self.Q
    
    @functools.cached_property
    def N(self):
        r"""
        c.f. pp 144 Ting, Elastic Anisotropy.
        
        fundamental elasticity matrix (Ingebrigtsen & Tonning, 1969)
        
        .. math::
            
            N = 
            \begin{bmatrix}
                N_1 & N_2 \\
                N_3 & N_1^T 
            \end{bmatrix}
        
        """
        if self._flag_N:
            self._N = np.column_stack((
                np.concatenate((self.N1, self.N2 )),
                np.concatenate((self.N3, self.N1.T))
                ))
            self._flag_N = 0
        return self._N
    
    @functools.cached_property
    def p(self):
        r"""
        eigenvalues (6,) of the fundamental elasticity matrix
        
        .. math::
            
            N \xi = p \xi
            
            N = \begin{bmatrix}
                    N_1 & N_2 \\
                    N_3 & N_1^T
                \end{bmatrix},
                
            \xi = \begin{bmatrix}
                    a \\ l
                  \end{bmatrix}
        
        Ref:
            Ting, T.C.T. (1996) Elastic Anisotropy. c.f. eqn. 5.5-3 pp. 144
        """
        if self._flag_eig:
            self._p, self._xi = LA.eig(self.N)
            self._flag_eig = 0
        return self._p
    
    @functools.cached_property
    def xi(self):
        r"""
        Right eigenvectors (6,6) of :math:`N \xi = p \xi`, see `p`.
        
        .. math::
            
            \xi = \begin{bmatrix}
                    a \\ l \\
                  \end{bmatrix}

        Ref:
            Ting, T.C.T. (1996) Elastic Anisotropy. c.f. eqn. 5.5-3 pp. 144
        """
        if self._flag_eig:
            self._p, self._xi = LA.eig(self.N)
            self._flag_eig = 0
        return self._xi # .round(tbx._PREC)
    
    @functools.cached_property
    def eta(self):
        r"""
        Left eigenvectors (6,6) of :math:`N^T \eta = p \eta`, see `p`.
                
        .. math::
            
            \eta = \begin{bmatrix}
                    l \\ a \\
                  \end{bmatrix}

        Ref:
            Ting, T.C.T. (1996) Elastic Anisotropy. c.f. eqn. 5.5-3 pp. 144
        """
        # return np.row_stack((self.l, self.a)) # "... the left eigenvector... are in the reverse order"""
        return (self.conI @ self.xi) # .round(tbx._PREC) # this is equivalent
        # return self.xi[::-1] # apparently Ting means the former, not reversal by index
    
    @functools.cached_property
    def a(self):
        r"""
        Stroh eigenvectors (6,6) solutions to the fundamental elasticity matrix
        The eigenvector `a` represents the direction of the displacement.
        """
        return self.xi[:3,:]
        # return np.concatenate((self.xi[:3,:], self.xi[:3,1::2]), axis=1)
        
    @functools.cached_property
    def l(self):
        r""" 
        Stroh eigenvectors (6,6) solutions to the fundamental elasticity matrix
        The eigenvector `l` represents the direction of traction.
        """
        return self.xi[3:,:]
    
    @functools.cached_property
    def A(self):
        r"""
        Stroh eigen vectors (3,3) obeying :math:`a_{\alpha+3} = \bar{a}_{\alpha}`
        (i.e. half the roots of the sextic equation)
        """
        return self.xi[:3, ::2]
    
    @functools.cached_property
    def L(self):
        r""" 
        Stroh eigen vectors (3,3) obeying :math:`l_{\alpha+3} = \bar{l}_{\alpha}`
        (i.e. half the roots of the sextic equation)
        """
        return self.xi[3:, ::2]
    
    @functools.cached_property
    def P(self):
        r""" 
        Stroh eigen values (3,) obeying :math:`p_{\alpha+3} = \bar{p}_{\alpha}`
        (i.e. half the roots of the sextic equation)
        """
        return self.p[::2]
    
    @functools.cached_property
    def M(self):
        r"""
        :math:`M_{\alpha i} L_{i \beta} = \partial _{\alpha \beta}`
        ref: Stroh (1958) Dislocations and cracks in anisotropic elasticity pp. 631
        """
        return LA.inv(self.L)
    
    @functools.cached_property
    def B(self):
        r"""
        :math:`B_{ij} = 1/2 i \sum_{\alpha}(A_{i \alpha} M_{\alpha j} - \bar{A}_{i \alpha} \bar{M}_{\alpha j})`
        ref: Stroh (1958) Dislocations and cracks in anisotropic elasticity pp. 631
        """
        return 0.5j * (self.A @ self.M - np.conjugate(self.A) @ np.conjugate(self.M))
    
    # FIXME These letters from Ting clash with the nomenclature of Stroh
    # --- from Ting ch. 5
# =============================================================================
#     @functools.cached_property
#     def S(self):
#         r"""
#         Barnett-Lothe tensors
#         """
#         return 1j * (2*self.A @ self.B.T - np.eye(3))
# 
#     @functools.cached_property
#     def H(self):
#         r"""
#         Barnett-Lothe tensors
#         """
#         return 2j * self.A @ self.A.T
#     
#     @functools.cached_property
#     def L(self):
#         r"""
#         Barnett-Lothe tensors
#         """
#         return -2j * self.L @ self.L.T
# =============================================================================
    
    # End Stroh
    

