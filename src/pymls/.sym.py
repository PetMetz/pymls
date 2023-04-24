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

class Symmetry():
    """ Symmetry operator class """
    _TOL = 1e-06
    
    def __add__(self, other:Symmetry) -> Symmetry:
        """ Operates only on translation """
        if isinstance(other, Symmetry):
            B = other.A
        else: # FIXME should check for array type
            B = other
        A = self.A.copy()
        A += B[:,-1]
        return Symmetry(A, self.origin)
        
    def __mul__(self, other:Symmetry) -> Symmetry:
        """ Operates only on rotation. """
        if isinstance(other, Symmetry):
            B = other.A
        else: # FIXME should check for array type
            B = other
        A = self.A.copy()
        A[:3,:3] = A[:3,:3] @ B[:3,:3]
        return Symmetry(A, self.origin)
    
    def __eq__(self, other) -> bool:
        return np.allclose(self.A, other.A, atol=self._TOL)
    
    def __str__(self):
        return f'<Symmetry(\nR:\n{self.R}\nT:\n{self.T}\n) @ {hex(id(self))}>'
        
    def __repr__(self):
        return str(self)
    
    def __init__(self, A:np.ndarray=None, origin:np.ndarray=None) -> None:
        """ A (4,4) augmented transformation matrix """
        self.A = A
        self.origin = origin
    
    def __call__(self, pts:np.ndarray) -> np.ndarray:
        """ apply `self` to `pt` in `pts`. """
        o = self.origin
        # operate on points
        pts = np.copy(pts)
        pts = np.reshape(pts, (-1, 3))
        pts -= o
        pts = np.column_stack((pts, np.ones(len(pts))))
        pts = np.einsum('ij,kj->ki', self.A, pts)[:,:3] # ij (4,4) ; lm (-1, 4)
        # operate on origin
        # o = np.append(o, (1,)).reshape((-1,4))
        # o = np.einsum('ij,kj->ki', self.A, o)[0,:3]
        return pts + o
            
    # === CONSTRUCTORS === #
    @classmethod
    def identity (cls) -> Symmetry:
        """ """
        return Symmetry(cls._I(4))
    
    @classmethod
    def translation(cls, vector:np.ndarray) -> Symmetry:
        """ """
        X = cls._O(4)
        X[:,-1] = np.concatenate((vector, (1,)))
        return Symmetry(X)

    @classmethod
    def rotation(cls, vector:np.ndarray, angle:float, degree=True, origin:np.ndarray=None) -> Symmetry:
        """
        From Rodrigues' Formula
        https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
        """
        X = cls._I(4)
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
    
    @classmethod
    def reflection(cls, normal:np.ndarray, origin:np.ndarray=None) -> Symmetry:
        """ reflection at plane with normal `hkl` """
        X = cls._I(4)
        hkl3 = normal / LA.norm(normal) * np.ones((3,3))
        X[:3,:3] -= 2 * hkl3.T * hkl3
        return Symmetry(X, origin)

    @classmethod
    def inversion(cls, origin:np.ndarray=None) -> Symmetry:
        X = cls._I(4)
        X[:3,:3] *= -1
        return Symmetry(X, origin)

    @classmethod
    def rotoinversion(cls, vector:np.ndarray, angle:float, degree:bool, origin:np.ndarray=None) -> Symmetry:
        S1 = cls.rotation(vector, angle, degree, origin)
        S2 = cls.inversion(origin)
        return S1 * S2

    @classmethod
    def rotoreflection(cls, vector:np.ndarray, angle:float, degree:bool, origin:np.ndarray=None) -> Symmetry:
        S1 = cls.rotation(vector, angle, degree, origin)
        S2 = cls.reflection(vector, origin)
        return S1 * S2

    @classmethod
    def screw(cls, axis:np.ndarray, angle:float, degree=True, origin:np.ndarray=None) -> Symmetry:
        ...
    
    @classmethod
    def glide(cls, vector:np.ndarray, hkl:np.ndarray, origin:np.ndarray=None) -> Symmetry:
        ...
        
    
    # === PROPERTIES === #
    @property
    def A(self):
        return self._A
    
    @A.setter
    def A(self, X:np.ndarray) -> None:
        assert X.shape == (4,4), 'Transformation matrix must have shape (4,4).'
        self._A = X
        
    @property
    def R(self):
        """ Rotation matrix """
        return self.A[:3,:3]
    
    @property
    def T(self):
        """ translation vector """
        return self.A[:3,-1]
    
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
        
    # === METHODS === #
    @staticmethod
    def _I(l:int):
        return np.eye(l)
    
    @staticmethod
    def _O(l:int):
        return np.zeros((l,l))
    
    def operate(self, pts:np.ndarray):
        return self.__call__(pts)
    
    def orbit(self, pts:np.ndarray):
        """ while new points != initial points, operate """
        ...
    
        
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


    #%%
    from topotools.utils import plot_scatter_3d
    
    fig, ax = plot_scatter_3d([])
    ax.plot((1,1), (1,1), (-2,2), 'k:')
    
    for pt in pts2:
        ax.scatter(*pt, color='red')
    
    for pt in pts3:
        ax.scatter(*pt, color='blue')


    #%%
    # -43m
    # 1
    # 3 || 111
    # 2 || 100
    # -4 || 001
    # m || 110
    R3 = Symmetry.rotation((1,1,1), 120)
    R2 = Symmetry.rotation((1,0,0), 180)
    R4b = Symmetry.rotation((0,0,1), 90) * Symmetry.inversion()
    M110 = Symmetry.reflection((1,1,0))

    pt = (0.123, 0.123, 0.246) # (x,x,z); s.g. 217 wyk pos 24g
    pts = [pt,]
    
