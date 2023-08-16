# -*- coding: utf-8 -*-
"""

NB for this case we only really care about symmetric rank 4 elastic tensors

Created on Mon Dec 19 18:11:40 2022

@author: UT
"""

import numpy as np

SQ2 = np.sqrt(2)


class Voigt():
    """ mappings for voigt reduction """
    _ORDER = None
    
    def __repr__(self):
        """ pretty print Cij """
        ...
    
    def __init__(self, T:np.ndarray=None,
                       M:np.ndarray=None,
                       V:np.ndarray=None,
                       case=None,
                       sym=False
                       ):
        """
        Voigt reduction scheme for tensors ((4,) -> (2,)) and matrices ((2,)->(1,))
        and inversion scheme for vectors ((1,) -> (2,)) and matrices ((2,)->(4,)).
        
        Unless higher rank object is known completely, use constructors.
        """
        self.case = case # 'c' or 's'
        self.tensor = T # Tijkl
        self.matrix = M # Mij
        self.vector = V # Vi
        self.sym = sym
        
    # constructors
    @classmethod
    def from_tensor(cls, X:np.ndarray, case=None):
        return cls(T=X, case=case)
    
    @classmethod
    def from_matrix(cls, X:np.ndarray, case=None):
        ...
    
    @classmethod
    def from_vector(cls, X:np.ndarray, case=None):
        X = np.asarray(X)
        assert len(X.shape) == 1, f'Incorrect shape for vector: {X}'
        ...
        
    # funcs
    @staticmethod
    def get_order(X:np.ndarray):
        return int(len(X.shape))
    
    @staticmethod
    def contract_ijkl(i,j,k,l):
        """ Ting, Anisotropic Elasticity: Theory and Applications. (1996) eqn. 2.3-5b """
        i, j, k, l = map(int, (i,j,k,l))
        if i == j:
            a = i 
        else:
            a = 9 - i - j
        if k == l:
            b = k 
        else:
            b = 9 - k - l
        return a, b
    
    @staticmethod
    def contract_ij(i, j):
        """ Ting, Anisotropic Elasticity: Theory and Applications. (1996) eqn. 2.3-1 """
        i, j = map(int, (i,j))
        if i == j:
            return i 
        else:
            return 9 - i - j
    
    def to_mandel(self):
        """ return Mandel object """
        ...
    
    # properties
    @property
    def case(self):
        return self._case
    
    @case.setter
    def case(self, s: str):
        if s is None or (isinstance(s, str) and 'c' in s.lower()):
            self._case = 'c'
        elif isinstance(s, str) and 's' in s.lower():
            self._case = 's'
        else:
            raise Exception(f'Undefined case (c|s) in Voigt: {s}')
    
    @property
    def tensor(self):
        return self._tensor
    
    @tensor.setter
    def tensor(self, X:np.ndarray):
        if X is None:
            return
        X = np.asarray(X)
        assert len(X.shape) == 4, f'Incorrect shape for tensor: {X}'
        self._tensor = X
        self._ORDER = 4
    
    @property
    def T(self):
        return self.tensor
    
    @T.setter
    def T(self, X:np.ndarray):
        self.tensor = X

    @property
    def matrix(self):
        return self._matrix
    
    @matrix.setter
    def matrix(self, X:np.ndarray):
        if X is None:
            return
        X = np.asarray(X)
        assert len(X.shape) == 2, f'Incorrect shape for matrix: {X}'
        self._matrix = X
        self.ORDER = 2
    
    @property
    def M(self):
        return self.matrix
    
    @M.setter
    def M(self, X:np.ndarray):
        self.matrix = X

    @property
    def vector(self):
        return self._vector
    
    @vector.setter
    def vector(self, X:np.ndarray):
        if X is None:
            return
        X = np.asarray(X)
        assert len(X.shape) == 1, f'Incorrect shape for vector: {X}'
        self._vector = X
        self._ORDER = 1
    
    @property
    def V(self):
        return self.vector
    
    @V.setter
    def V(self, X:np.ndarray):
        self.vector = X


    

    # End of class Voigt

    
    
class Mandel():
    """ mappings for Mandel reduction """
    
    ...
    
    # end of class Mandel




#%%% quick and dirty tests

if __name__ == "__main__":
    
    v3    = np.array((1, 2, 3))
    m33   = np.outer(v3,v3)
    m99   = np.outer(m33,m33)
    t3333 = m99.reshape((3,3,3,3)) 
    
    v = Voigt.from_tensor(t3333, case='s')
