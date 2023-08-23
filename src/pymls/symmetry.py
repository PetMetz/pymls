#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

ref:
    https://www.iucr.org/__data/assets/pdf_file/0019/15823/22.pdf
    https://github.com/materialsproject/pymatgen/blob/v2023.2.28/pymatgen/core/operations.py
    
    
Created on Thu Mar  9 12:56:17 2023

@author: pcm
"""
from __future__ import annotations
from typing import Union
import functools
import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation


def set_origin(fn):
    @functools.wraps(fn)
    def dec(*args, **kwargs):
        if 'origin' in kwargs:
            pass
        else:
            kwargs.update({'origin':np.array((0,0,0))})
        return fn(*args, **kwargs)
    return dec


#%%
class Affine():
    """ base class for symmetry operations """
    _TOL = 0.1
    
    def __str__(self):
        return f'<Affine(\nM:\n{self.M}\nT:\n{self.T}\nO:{self.origin}\n) @ {hex(id(self))}>'
        
    def __repr__(self):
        return str(self)
    
    def __add__(self, other:Affine) -> Affine:
        """ Operates only on translation """
        B = np.asarray(other)
        A = self.A.copy()
        A[:3,-1] += B[:3,-1]
        return Symmetry(A, self.origin)
        
    def __mul__(self, other:Affine) -> Affine:
        """ Operates only on rotation. """
        B = np.asarray(other)
        A = self.A.copy()
        A[:3,:3] = A[:3,:3] @ B[:3,:3]
        return Symmetry(A, self.origin)
    
    def __matmul__(self, other:Affine) -> Affine:
        """ self dot other """
        return Symmetry(np.asarray(other) @ self.A)
    
    def __eq__(self, other) -> bool:
        return np.allclose(self.A, other.A, atol=self._TOL)

    def __getitem__(self, index):
        return self.A[index]
    
    def __setitem__(self, index, val):
        self.A[index] = val
        
    def __array__(self):
        return self.A

    def __call__(self, pts:np.ndarray) -> np.ndarray:
        """ apply `self` to `pt` in `pts`. """
        o = self.origin
        # operate on points
        pts = np.copy(pts).astype(float)
        pts = np.reshape(pts, (-1, 3))
        pts -= o
        pts = np.column_stack((pts, np.ones(len(pts))))
        pts = np.einsum('ij,kj->ki', self.A, pts)[:,:3] # ij (4,4) ; lm (-1, 4)
        return pts + o

    def __init__(self, A:np.ndarray=None, origin:np.ndarray=None) -> None:
        """ A (4,4) augmented transformation matrix """
        self.A = A
        self.origin = origin
        
    @property
    def A(self):
        return self._A
    
    @A.setter
    def A(self, X:Union[Affine,np.ndarray]) -> None:
        if X is None:
            X = np.eye(4)
        X = np.asarray(X)
        assert X.shape == (4,4), 'Transformation matrix must have shape (4,4).'
        self._A = X
        
    @property
    def M(self):
        """ Rotation matrix """
        return self._A[:3,:3]
    
    @M.setter
    def M(self, X:np.ndarray):
        self._A[:3,:3] = X
    
    @property
    def T(self):
        """ translation vector """
        return self._A[:3,-1]

    @T.setter
    def T(self, X:np.ndarray):
        self._A[:3,-1] = X

    @property
    def origin(self):
        return self._origin
    
    @origin.setter
    def origin(self, X:np.ndarray):
        if X is None:
            self._origin = np.array((0,0,0))
        else:
            X = np.asarray(X)
            assert X.shape == (3,), 'Origin must be (3,) vetor.'
            self._origin = X

    @property
    def inverse(self):
        return Affine(LA.inv(self.A))
    
    # End of Affine
    

class Symmetry(Affine):
    """ Symmetry operator class """
    
    def __str__(self):
        return f'<Symmetry(\nR:\n{self.M}\nT:\n{self.T}\n) @ {hex(id(self))}>'
        
    def __repr__(self):
        return str(self)
    
    def __init__(self, A:np.ndarray=None, origin:np.ndarray=None) -> None:
        """ A (4,4) augmented transformation matrix """
        super().__init__(A, origin)
    
    # === CONSTRUCTORS === #
    @staticmethod
    def identity() -> Symmetry:
        """ """
        return Symmetry(np.eye(4))
    
    @staticmethod
    def translation(vector:np.ndarray) -> Symmetry:
        """ """
        X = np.eye(4)
        X[:,-1] = np.concatenate((vector, (1,)))
        return Symmetry(X)

    @staticmethod
    def rotation(vector:np.ndarray, angle:float, degree=True, origin:np.ndarray=None) -> Symmetry:
        """
        From Rodrigues' Formula
        https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
        """
        X = np.eye(4)
        if degree:
            angle = angle * np.pi / 180 # as radian
        I = np.eye(len(vector))
        u = vector / np.linalg.norm(vector) # as unit vector
        uu = np.outer(u,u)
        ux = np.cross(u, -I) # https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
        cos = np.cos(angle)
        sin = np.sin(angle)
        X[:3,:3] = cos * I + sin * ux + (1-cos) * uu
        return Symmetry(X, origin)
    
    @staticmethod
    def reflection(normal:np.ndarray, origin:np.ndarray=None) -> Symmetry:
        """ reflection at plane with normal `hkl` """
        X = np.eye(4)
        hkl3 = normal / LA.norm(normal) * np.ones((3,3))
        X[:3,:3] -= 2 * hkl3.T * hkl3
        return Symmetry(X, origin)

    @staticmethod
    def inversion(origin:np.ndarray=None) -> Symmetry:
        X = np.eye(4)
        X[:3,:3] *= -1
        return Symmetry(X, origin)

    @staticmethod
    def rotoinversion(vector:np.ndarray, angle:float, degree:bool, origin:np.ndarray=None) -> Symmetry:
        S1 = Symmetry.rotation(vector, angle, degree, origin)
        S2 = Symmetry.inversion(origin)
        return S1 * S2

    @staticmethod
    def rotoreflection(cls, vector:np.ndarray, angle:float, degree:bool, origin:np.ndarray=None) -> Symmetry:
        S1 = Symmetry.rotation(vector, angle, degree, origin)
        S2 = Symmetry.reflection(vector, origin)
        return S1 * S2

    @staticmethod
    def screw(cls, axis:np.ndarray, angle:float, degree=True, origin:np.ndarray=None) -> Symmetry:
        ...
    
    @staticmethod
    def glide(cls, vector:np.ndarray, hkl:np.ndarray, origin:np.ndarray=None) -> Symmetry:
        ...
        
    
    # === PROPERTIES === #

        
    # === METHODS === #
    def operate(self, pts:np.ndarray):
        return self.__call__(pts)
    
    def orbit(self, pts:np.ndarray):
        """ while new points != initial points, operate """
        ...
    
    # End of Symmetry
    
        
#%%
if __name__ == "__main__":
    pt = np.array((2,2,2))
    SO = Symmetry.rotation((0,0,1), 90) * Symmetry.inversion()
    
    pts = [pt,]
    for _ in range(10):
        npt = SO(pts[-1])[0] # mutate last point
        if all(np.isclose(npt,pts[0])): # if orbit complete, end
            break
        pts.append(npt)
        
        
    A = SO.A
    pts2 = [pt,]
    for _ in range(10):
        npt = np.concatenate((pts2[-1], (1,)))
        npt = np.inner(A, npt)[:3]
        if all(np.isclose(npt, pts2[0])): # if orbit complete, end
            break
        pts2.append(npt)

    assert np.all(np.isclose(pts, pts2)), 'Error in indexing of einum'
    

    pts3 = [(2,2,2)]
    SO = Symmetry.rotation((0,0,1), 90, origin=(1,1,1)) * Symmetry.inversion()
    for _ in range(100):
        npt = SO(pts3[-1])[0] # mutate last point
        if all(np.isclose(npt, pts3[0])): # if orbit complete, end
            break
        pts3.append(npt)


    SO = Symmetry.reflection((0,0,1), origin=(0,0,-2))
    pts4 = np.concatenate((SO(pts2), SO(pts3)))


