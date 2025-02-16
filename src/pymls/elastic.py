 # -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:40:09 2022

@author: pmetz1
"""
# built-ins
import functools
from typing import Union
from collections import namedtuple

# 3rd party
import numpy as np
from numpy import linalg as LA
import scipy.linalg as sla

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
    (66, 66, 66, 66, 66, 99, 99, 99, 99, 44, 44)
    ), dtype=int)

_QZSOLUTION = namedtuple('qz_solution', ['alpha', 'beta', 'vl', 'vr', 'nvl', 'nvr', 'work', 'info', 'S', 'T', 'Q', 'Z', 'eigv'])

# - conventions
_voigt3 = np.array(((0,0), (1,1), (2,2), (1,2), (0,2), (0,1)), dtype=int)
_voigt6 = np.array([ np.concatenate((np.ones_like(_voigt3)*_voigt3[i], _voigt3), axis=1) for i in range(6) ], dtype=int)


# --- functions
def generate_index(arg:int) -> tuple:
    """ 
    generate indices from `_ELASTIC_RESTRICTIONS`.
    
    Returns
    -------
    tuple (int, int)
        ## -> (#, #)
    """
    i, j = divmod(abs(arg), 10)
    return tuple((i-1, j-1))


def generate_indices(arg:np.ndarray) -> np.ndarray:
    """ See `generate_index`. """
    return np.apply_along_axis(generate_index, axis=0, arr=arg)


def get_unique(LaueGroup:int) -> tuple:
    """ Unique symmetry invariant indices for the corresponding `LaueGroup`. """
    rv = _ELASTIC_RESTRICTIONS[:, LaueGroup-1] # NB zero index
    return np.unique(abs(rv[(rv!=0) & (rv!=99)]))


def parse_laue_class(arg:Union[str,int]) -> int:
    """
    Identify Laue class by integer (1-11), crystal system (cubic, ...) or
    point group (m-3m, ...) and return corresponding integer.

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


# FIXME factor out for / if with masking?
def cij_from_group(*cij, group:Union[str,int]) -> np.ndarray:
    """
    Cij matrix formed from unique elements corresponding to the Laue group.
    See `cij_order(LaueGroup)` for expected order.
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


def cij_from_vector(group, *cij) -> np.ndarray:
    """ Alias `cij_from_group`. """
    return cij_from_group(*cij, group=group)


def cij_from_dict(group, **cij) -> np.ndarray:
    """
    Cij matrix formed from unique elements corresponding to the Laue group.
    `ij` pairs are passed explicitly.
    See `cij_order(LaueGroup)` for expected pairs.
    """
    order = cij_order(group)
    elements = np.asarray(list(cij.values()), dtype=float)
    keys = np.asarray(list(cij.keys()), dtype=int)
    m = np.argwhere(np.in1d(order, keys))
    return cij_from_group(*elements[m], group=group)


def cij_order(group) -> np.ndarray:
    """ Order of `ij` pairs used to construct Cij from vector. """
    laue = parse_laue_class(group)
    return get_unique(laue)


def sijkl_from_cijkl(cijkl):
    """
    :math:`cijkl sijkl = 1/2 (\delta_{im} \delta_{jn} + \delta_{in} \delta_{jm}`.
    Armstring & Lynch (2004) in Diffraction Analysis of the Microstructure of
       Materials (Mittemeijer & Scardi, eds.)
    """
    indices = np.indices(cijkl.shape)
    indices = indices.T.reshape((-1,4))
    i, j, k, l = indices.T
    dik = (i==k).astype(int)
    djl = (j==l).astype(int)
    dil = (i==l).astype(int)
    djk = (j==k).astype(int)
    A = 1/2 * (dik*djl + dil*djk).reshape(cijkl.shape)
    ... 

# --- classes
class Stroh():
    """
    Elastic aspects of the dislocation contrast factor obtained by solution to
    the Stroh formulation.

    Reference
    ---------
        Ting, T.C.T. (1996) Elastic Anisotropy. ISBN 9780195074475
    """
    # - class state
    _flag_N = 1
    _flag_eig = 1

    @classmethod
    def reset(cls) -> None:
        """ Reset class state. """
        cls._flag_N = 1
        cls._flag_eig = 1

    # - overloads
    def __repr__(self) -> str:
        return f'<Stroh(cij=\n{self.cij}, crystalSystem={self.crystalsystem}) @ {hex(id(self))}>\nP_i = {self.P.round(3)}'

    def __init__(self,
                 cij:np.ndarray=None,
                 crystalSystem:str=None,
                 dislocation=None
                 ) -> None:
        """
        Elastic aspects of the dislocation contrast factor obtained by solution to
        the Stroh formulation.

        Parameters
        ----------
        cij : np.ndarray, optional
            DESCRIPTION. The default is None.
        crystalSystem : str, optional
            DESCRIPTION. The default is None.
        dislocation : ..., optional
            Description. The default is None.

        Reference
        ---------
        Ting, T.C.T. (1996) Elastic Anisotropy. ISBN 9780195074475
        """
        self.cij = cij
        self.crystalsystem = crystalSystem
        self.dislocation = dislocation

    # FIXME apply state flags when setter is called
    @property
    def cijkl(self) -> np.ndarray:
        """ Elastic stiffness tensor. """
        return self._cijkl

    @cijkl.setter
    def cijkl(self, X) -> None:
        self._cijkl = X

    @property
    def cij(self) -> np.ndarray:
        """ Reduced stiffness matrix. """
        return self.apply_mandel(self._cijkl)
    
    @cij.setter
    def cij(self, X) -> None:
        if not X is None:
            self._cijkl = self.invert_mandel(X)

    # FIXME This differs from the treatment by Ting, not sure it's necessary
    @functools.cached_property
    def tcijkl(self) -> np.ndarray:
        r"""
        Martinez-Garcia, PHYSICAL REVIEW B 76, 174117, 2007
        """
        H = self.cij[0,0] - self.cij[0,1] - 2*self.cij[3,3]
        c12 = self.cij[0,1]
        c44 = self.cij[3,3]
        P = LA.inv(self.dislocation.P) # transform of {e1, e2, e3} into {x1, x2, x3}
        dij = lambda i, j: 1 if i==j else 0
        I = np.indices(self.cijkl.shape).T.reshape((-1,4))
        rv = np.zeros(self.cijkl.shape)
        for index in I:
            i,j,k,l = index
            rv[tuple(index)] = c12*dij(i,j)*dij(k,l) + c44*dij(i,j)*dij(k,l) +\
                H * np.sum([np.prod((P[r,i],P[r,j],P[r,k],P[r,l])) for r in range(3)])
        return rv

    @functools.cached_property
    def tcij(self) -> np.ndarray:
        return tbx.contract_ijkl(self.tcijkl)

    def apply_voigt(self, X) -> np.ndarray:
        """
        Apply Voigt reduction scheme to X(3,3) to produce X2(6,).
        11 -> 1; 22 -> 2; 33 -> 3; 23 -> 4; 13 -> 5; 12 -> 6
        """
        return np.array([X[tuple(e)] for e in _voigt3])

    def invert_voigt(self, X) -> np.ndarray:
        """
        Given a Voigt reduced vector, reconstruct 2nd order tensor.
        """
        a = np.zeros((3,3))
        for idx, pt in enumerate(_voigt3):
            a[tuple(pt)] = X[idx]
            a[tuple(pt[::-1])] = X[idx]
        return a

    def apply_mandel(self, X) -> np.ndarray:
        """
        The extension of Voigt reduction to the 4th rank tensor representing
        proportionality of two 2nd rank tensors.
        """
        rv = np.zeros((6,6))
        for ijkl in _voigt6.reshape((-1,4)):
            try:
                ij = tbx.map_ijkl(*ijkl)
                rv[ij] = X[tuple(ijkl)]
            except IndexError:
                print(f'Warning: unable to set {ijkl} -> {ij}')
        return rv

    # FIXME can't this be vectorized?
    def invert_mandel(self, X, case='c') -> np.ndarray:
        """
        Given a Voigt reduced 2nd order tensor, reconstruct 4th order tensor.
        """
        a = np.zeros((3,3,3,3))
        # - form index
        I = self.elastic_symmetry
        I0 = I[0]
        # - apply mapping
        for ijkl in I0:
            mn = tbx.map_ijkl(*ijkl)
            a[tuple(ijkl)] = X[tuple(mn)]
        # - exhaustive equivalence
        for Ip in I[1:]:
            for idx, jdx in zip(I0, Ip):
                a[tuple(jdx)] = a[tuple(idx)]
        return a

    @functools.cached_property
    def elastic_symmetry(self) -> np.ndarray:
        """ form symmetry equivalents ijks = jiks = ksij = ijsk """
        shape = (3,3,3,3)
        I = np.transpose(np.indices(shape)).astype(int) # 3, 3, 3, 3, 4
        I = I.reshape((-1, len(shape))) # -1, 4
        rv = []
        # form symmetry equivalents ijks = jiks = ksij = ijsk
        rv.append( I[:, (0,1,2,3)] ) # ijks
        rv.append( I[:, (1,0,2,3)] ) # jiks
        rv.append( I[:, (2,3,0,1)] ) # ksij
        rv.append( I[:, (0,1,3,2)] ) # ijsk
        return rv

    # FIXME can't this be vectorized?
    def apply_elastic_symmetry(self, X)  -> np.ndarray:
        I0, *I = self.elastic_symmetry
        # - exhaustive equivalence
        for Ip in I[1:]:
            for idx, jdx in zip(I0, Ip):
                X[tuple(jdx)] = X[tuple(idx)]
        return X

    @property
    def crystalsystem(self) -> str:
        print('Warning: crystal symmetry not yet implemented')
        return self._crystalsystem

    @crystalsystem.setter
    def crystalsystem(self, x:str) -> None:
        self._crystalsystem = x

    # FIXME incomplete
    def apply_crystal_symmetry(self, group):
        """ """
        ...

    # @functools.cached_property
    @property
    def Q(self) -> np.ndarray:
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
        # idx = [(0,0), (0,5), (0,4),
        #        (0,5), (5,5), (4,5),
        #        (0,4), (4,5), (4,4)] # NB zero index
        # return np.reshape([self.cij[e] for e in idx], (3,3))
        return self.cijkl[:,0,:,0]

    # @functools.cached_property
    @property
    def R(self) -> np.ndarray:
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
        # idx = [(0,5), (0,1), (0,3),
        #        (5,5), (1,5), (3,5),
        #        (4,5), (1,4), (3,4)] # NB zero index
        # return np.reshape([self.cij[e] for e in idx], (3,3))
        return self.cijkl[:,0,:,1]

    # @functools.cached_property
    @property
    def T(self) -> np.ndarray:
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
        # idx = [(5,5), (1,5), (3,5),
        #        (1,5), (1,1), (1,3),
        #        (3,5), (1,3), (3,3)] # NB zero index
        # return np.reshape([self.cij[e] for e in idx], (3,3))
        return self.cijkl[:,1,:,1]

    @functools.cached_property
    def _gA(self) -> np.ndarray:
        """
        Left matrix of the generalized eigenproblem (Ting 5.5-1)
        e.g. for qz decomposition
        """
        return tbx.square([(-self.Q,  tbx.O),
                           (-self.R.T, tbx.I)
                           ])
    
    @functools.cached_property
    def _gB(self) -> np.ndarray:
        """
        Right matrix of the generalized eigenproblem (Ting 5.5-1)
        e.g. for qz decomposition
        """
        return tbx.square([(self.R, tbx.I),
                           (self.T, tbx.O)
                          ])
    
    @functools.cached_property
    def qz(self):
        """
        Eigenvalue solution by method of qz decomposition, and ordered such that
        conjugate pairs occur [a1, a2, a3, a1*, a2*, a3*].
        
        References:
            https://www.netlib.org/lapack/lug/node56.html
            https://www.netlib.org/lapack/lug/node35.html#1803
            https://stackoverflow.com/questions/49007201/scipy-qz-generalized-eigenvectors
        """
        S, T, Q, Z = sla.qz(self._gA, self._gB, output='complex') # this is redundant
        alpha, beta, vl, vr, work, info = sla.lapack.zggev(self._gA, self._gB)
        eigv = alpha / beta # == Sii / Tii
        nvl = vl / LA.norm(vl, axis=0)
        nvr = vr / LA.norm(vr, axis=0)
        m = np.argsort(-np.sign(eigv.imag)) # I'm not sure why this maintains ordering, but it appears to up to triclinic
        return _QZSOLUTION(alpha=alpha, beta=beta, work=work, info=info, S=S,
                           T=T, Q=Q, Z=Z, eigv=eigv[m],
                           vl=vl[:,m], vr=vr[:,m],
                           nvl=nvl[:,m], nvr=nvr[:,m])
                           # eigv=eigv, vl=vl, vr=vr, nvl=nvl, nvr=nvr)
    @functools.cached_property
    def N1(self) -> np.ndarray:
        r"""
        c.f. pp 144 Ting, Elastic Anisotropy.

        .. math::
            N_1 = -T^{-1} R^T \ \  [-]
        """
        return -LA.inv(self.T) @ np.transpose(self.R)

    @functools.cached_property
    def N2(self) -> np.ndarray:
        r"""
        c.f. pp 144 Ting, Elastic Anisotropy.

        .. math::
            N_2 = T^{-1} \ \ [m^2 \cdot N^{-1}]
        """
        return LA.inv(self.T)

    @functools.cached_property
    def N3(self) -> np.ndarray:
        r"""
        c.f. pp 144 Ting, Elastic Anisotropy.

        .. math::
            N_3 = R T^{-1} R^T - Q \ \ [N \cdot m^{-2}]
        """
        return self.R @ LA.inv(self.T) @ np.transpose(self.R) - self.Q

    @functools.cached_property
    def N(self) -> np.ndarray:
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
            self._N = tbx.square((
                (self.N1, self.N2  ),
                (self.N3, self.N1.T)
                ))
            self._flag_N = 0
        return self._N

    def _get_eig(self) -> None:
        if self._flag_eig:
            # order = [0, 1, 2, 3, 4, 5] # unordered
            order = [0, 2, 4, 1, 3, 5] # ordered imag(p_alphha) > 0
            # order = [1, 3, 5, 0, 2, 4] # ordered imag(p_alphha) < 0
            w, vl, vr = sla.eig(self.N, left=True, right=True) # The normalized (unit "length") eigenvectors, such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
            self._p   = w[order] # numpy returns ordered pairs, Ting shows ordered conjugates
            self._eta = vl[:, order]
            self._xi  = vr[:, order]
            self._flag_eig = 0        

    @functools.cached_property
    def p(self) -> np.ndarray:
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
            self._get_eig()
        return self._p
        # return self.qz.eigv

    @functools.cached_property
    def xi(self) -> np.ndarray:
        r"""
        Right eigenvectors (6,6) of :math:`N \xi = p \xi`, see `p`.
        NB scipy.linalg.eig returns column eigenvectors
        
        .. math::

            \xi = \begin{bmatrix}
                    a \\ l \\
                  \end{bmatrix}
                  
                =
                 
            \begin{bmatrix}
                a_{11} & a_{21} &  a_{31} & \bar{a_{11}} & \bar{a_{21}} & \bar{a_{31}} \\
                a_{12} & a_{22} &  a_{32} & \bar{a_{12}} & \bar{a_{22}} & \bar{a_{32}} \\
                a_{13} & a_{23} &  a_{33} & \bar{a_{13}} & \bar{a_{23}} & \bar{a_{33}} \\ 
                l_{11} & l_{21} &  l_{31} & \bar{l_{11}} & \bar{l_{21}} & \bar{l_{31}} \\
                l_{12} & l_{22} &  l_{32} & \bar{l_{12}} & \bar{l_{22}} & \bar{l_{32}} \\
                l_{13} & l_{23} &  l_{33} & \bar{l_{13}} & \bar{l_{23}} & \bar{l_{33}} \\
            \end{bmatrix}
                  

        Ref:
            Ting, T.C.T. (1996) Elastic Anisotropy. c.f. eqn. 5.5-3 pp. 144
        """
        if self._flag_eig:
            self._get_eig()
        return self._xi # .round(tbx._PREC)
        # return self.qz.vr

    # FIXME eta* = conI @ xi (instead of eta)
    @functools.cached_property
    def eta(self) -> np.ndarray:
        r"""
        Left eigenvectors (6,6) of :math:`N^T \eta = p \eta`, see `p`.

        .. math::

            \eta = \begin{bmatrix}
                    l \\ a \\
                  \end{bmatrix}

        Ref:
            Ting, T.C.T. (1996) Elastic Anisotropy. c.f. eqn. 5.5-3 pp. 144
        """
        # return tbx.conI @ self.xi # .round(tbx._PREC) # this is equivalent
        if self._flag_eig:
            self._get_eig()
        return self._eta # .round(tbx._PREC)
        # return self.qz.vl

    @functools.cached_property
    def a(self) -> np.ndarray:
        r"""
        Stroh eigenvectors (3,6) solutions to the fundamental elasticity matrix
        obeying
        
        .. math::
            
            a_{\alpha+3} = \bar{a_{\alpha}} 
            
        The eigenvector `a` represents the direction of the displacement.
        

        Returns
        -------
        np.ndarray (3,6) imaginary
            Half the redundant right Stroh eigenvectors
        
        Reference
        ---------
        c.f. Ting eqn. 5.5-4 & 5.3-11
        """
        # return self.xi[:3, (0,2,4,1,3,5)] # ordered
        return self.xi[:3,:] # / LA.norm(self.xi[:3,:], axis=0) # paired

    @functools.cached_property
    def l(self) -> np.ndarray:
        r"""
        Stroh eigenvectors (3,6) solutions to the fundamental elasticity matrix
        obeying
        
        .. math::
            
            b_{\alpha+3} = \bar{b_\alpha}
        
        The eigenvector `l` represents the direction of traction.


        Returns
        -------
        np.ndarray (3,6) imaginary
            Half the redundant right Stroh eigenvectors
        
        
        Reference
        ---------
        c.f. Ting eqn. 5.5-4 & 5.3-11
        """
        # return self.xi[3:, (0,2,4,1,3,5)] # ordered
        return self.xi[3:,:] # / LA.norm(self.xi[3:,:], axis=0) # paired

    @functools.cached_property
    def A(self) -> np.ndarray:
        r"""
        Stroh eigen vectors (3,3) obeying :math:`a_{\alpha+3} = \bar{a}_{\alpha}`
        (i.e. half the roots of the sextic equation).
        The eigenvector `A` represents the direction of displacement.
        :math:`A = [a1, a2, a3]` (NB column vectors)
        
        NB the degenerate vector `a` is normalized, the half-vector A is not.
        
        Returns
        -------
        np.ndarray (3,3) imaginary
            Unique half of the right Stroh eigenvectors.
        
        
        Reference
        ---------
        c.f. Ting eqn. 5.5-4 & 5.3-11
        """
        # return self.xi[:3, ::2] # == self.a[:, ::2]
        return self.xi[:3, :3] #  / LA.norm(self.xi[:3, :3], axis=0)  # ordered

    @functools.cached_property
    def L(self) -> np.ndarray:
        r"""
        Stroh eigen vectors (3,3) obeying :math:`l_{\alpha+3} = \bar{l}_{\alpha}`
        (i.e. half the roots of the sextic equation).
        The eigenvector `L` represents the direction of traction.
        :math:`B = [b1, b2, b3]` (NB column vectors)
        
        NB the degenerate vector `l` is normalized, the half-vector L is not.
        
        Returns
        -------
        np.ndarray (3,3) imaginary
            Unique half of the right Stroh eigenvectors.
        
        
        Reference
        ---------
        c.f. Ting eqn. 5.5-4 & 5.3-11
        """
        # return self.xi[3:, ::2] # == self.l[:, ::2]
        return self.xi[3:, :3] #  / LA.norm(self.xi[3:, :3], axis=0) 
       
    @functools.cached_property
    def P(self) -> np.ndarray:
        r"""
        Stroh eigen values (3,) obeying :math:`p_{\alpha+3} = \bar{p}_{\alpha}`
        (i.e. half the roots of the sextic equation) with positive imaginary
        component.
        
        
        Returns
        -------
        np.ndarray (3,) imaginary
            Unique half of the right Stroh eigenvalues. 
        
        
        Reference
        ---------
        ...
        """
        # return self.p[::2]
        return self.p[:3]

    @functools.cached_property
    def M(self) -> np.ndarray:
        r""" 
        Impedance tensor `M`
        """
        return -1j * self.L @ LA.inv(self.A)

# =============================================================================
#     @functools.cached_property
#     def M(self) -> np.ndarray:
#         r"""
#         :math:`M_{\alpha i} L_{i \beta} = \partial _{\alpha \beta}`
#         
#         
#         Returns
#         -------
#         np.ndarray (3,3) imaginary
#         
#         
#         Reference
#         ---------
#         Stroh (1958) Dislocations and cracks in anisotropic elasticity pp. 631
#         """
#         return LA.inv(self.L)
# 
#     @functools.cached_property
#     def B(self) -> np.ndarray:
#         r"""
#         :math:`B_{ij} = 1/2 i \sum_{\alpha}(A_{i \alpha} M_{\alpha j} - \bar{A}_{i \alpha} \bar{M}_{\alpha j})`
# 
# 
#         Returns
#         -------
#         np.ndarray (3,3) imaginary
# 
# 
#         Reference
#         ---------
#         Stroh (1958) Dislocations and cracks in anisotropic elasticity pp. 631
#         """
#         return (0.5j * (self.A @ self.M - np.conjugate(self.A) @ np.conjugate(self.M))).real
# =============================================================================

    # End Stroh


