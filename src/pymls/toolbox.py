# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 18:10:36 2022

@author: UT
"""
# built-ins
import functools

# 3rd party
import numpy as np
from numpy import linalg as LA
from scipy import spatial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
# from tqdm import tqdm

_SMALL = 1e-6
_PREC  = 9


def abt(a: np.ndarray, b: np.ndarray, degrees=False):
    rv = np.arccos(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    if degrees:
        return 180 / np.pi * rv
    return rv


def square(X):
    """ 
    X = [(A1,..., AN), ... , (B1, ..., BN)] -> X(NM, NM)
    i.e. [(row_1), (row_2), ...]
    """
    rows = list(map(np.column_stack, X))
    return np.row_stack(rows)


def float_tol(a, b, sig=None):
    sig = sig or _SMALL
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return all(abs(a - b).ravel() < sig)


def complex_tol(a, b, sig=None):
    sig = sig or _SMALL
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    d = (a - b).ravel()
    m = np.sqrt(d * np.conjugate(d)).real
    return all(m < sig)


def vol_from_scalar(a,b,c,al,be,ga):
    cosa = np.cos(al*np.pi/180)
    cosb = np.cos(be*np.pi/180)
    cosg = np.cos(ga*np.pi/180)
    return a*b*c*np.sqrt(1 + 2*cosa*cosb*cosg - cosa**2 - cosb**2 - cosg**2)


def all_unit_vectors(x) -> bool:
    return all( np.apply_along_axis(is_unit_vector, axis=1, arr=x) )

    
def is_orthogonal(X:np.ndarray) -> bool:
    """ det| X(N,N) | == 1 """
    return abs(LA.det(X)**2 - 1) <= _SMALL


def is_unit_vector(x: np.ndarray) -> bool:
    r""" :math:`\left|\vec{x}\right| == 1` """
    return abs(np.linalg.norm(x) - 1) <= _SMALL


def is_symmetric(X: np.ndarray) -> bool:
    """ X == X.T """
    return float_tol(X, X.T)


def orthogonal(fn):
    @functools.wraps(fn)
    def dec(*args, **kwargs):
        rv = fn(*args, **kwargs)
        if abs(LA.det(rv)**2 - 1) >= _SMALL:
            n  = fn.__name__
            print(f'Warning: {n} not orthogonal with det|{n}| = {LA.det(rv):.6f}')
        return rv
    return dec


def unit_vectors(fn):
    @functools.wraps(fn)
    def dec(*args, **kwargs):
        rv = fn(*args, **kwargs)
        test = np.apply_along_axis(LA.norm, axis=1, arr=rv)
        if any(abs(test - 1) >= _SMALL):
            n  = fn.__name__
            print(f'Warning: {n} not properly normalized. |ei| = {test[0]:.6f} {test[1]:.6f} {test[2]:.6f}') 
        return rv
    return dec


# FIXME this could just be a case of unit_vectors
def unit_vector(fn):
    @functools.wraps(fn)
    def dec(*args, **kwargs):
        rv = fn(*args, **kwargs)
        test = LA.norm(rv)
        if abs(test - 1) >= _SMALL:
            n = fn.__name__
            print(f'Warning: {n} not properly normalized. |v| = {test:.6f}')
        return rv
    return dec


def get_largest(X):
    a = np.apply_along_axis(np.linalg.norm, 0, X)
    return max(a)


def get_smallest(X):
    a = np.apply_along_axis(np.linalg.norm, 0, X)
    return min(a)


def generate_hull(matrix, o=None):
    # - defaults
    if o is None:
        o = np.zeros(3)
    # - generate points
    corners = np.array((
                        o,
                        o+(1,0,0),
                        o+(1,1,0),
                        o+(0,1,0),
                        o+(0,0,1),
                        o+(1,0,1),
                        o+(1,1,1),
                        o+(0,1,1),
                        ))
    corners = np.transpose(matrix @ corners.T)
    return spatial.ConvexHull(corners)


# FIXME: this is rather inelegant
def plot_cell(L, o=None, ax=None):
    # - defaults
    if o is None:
        o = np.array((0,0,0))
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect((1,1,1))
    elif not ax is None:
        fig = ax.get_figure()
    # - generate points
    corners = np.array((
                        o,
                        o+(1,0,0),
                        o+(1,1,0),
                        o+(0,1,0),
                        o+(0,0,1),
                        o+(1,0,1),
                        o+(1,1,1),
                        o+(0,1,1),
                        ))
    corners = np.transpose(L.matrix.T @ corners.T)
    index = np.arange(len(corners))
    # - brute force
    segments = []
    for idx, pt1 in enumerate(corners):
        row = []
        for pt2 in corners[index != idx]:
            row.append((pt1, pt2))
        segments.append(row)
    segments = np.array(segments).reshape(-1, 2, 3)
# =============================================================================
#     # - min dist fails for non orthogonal cells, float point issues
#     segments = []
#     for idx, pt1 in enumerate(corners):
#         x = np.linalg.norm(corners[index!=idx]-pt1, axis=1)
#         m = np.argwhere(abs(x-min(x))<=1e-6).reshape(-1)
#         for pt2 in corners[m]:
#             segments.append((pt1, pt2))
#     segments = np.array(segments)
# =============================================================================
# =============================================================================
#     # - I apparently don't know how convex hulls work
#     hull = spatial.ConvexHull(corners)
#     segments = []
#     for simplex in hull.simplices:
#         sub = corners[simplex]
#         segments.append((sub[0], sub[1]))
#         segments.append((sub[1], sub[2]))
#     segments = np.array(segments)
# =============================================================================
    # - add to figure as collection
    collection = Line3DCollection(segments, color='k', alpha=0.5, linestyle=':')
    ax.add_collection(collection)
    plt.show()
    # - adjustments
    minl = np.min(np.min(corners, axis=0))
    maxl = np.max(np.max(corners, axis=0))
    ax.set_xlim(minl, maxl)
    ax.set_ylim(minl, maxl)
    ax.set_zlim(minl, maxl)
    return fig, ax
    

def plot_r3(a, b, c, o=None, ax=None, labels=None):
    # - defaults
    if labels is None:
        labels = ('x1', 'x2', 'x3')
    if o is None:
        o = np.array((0,0,0))
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect((1,1,1))
        # - calcs
        maxl = get_largest(np.transpose((o,a,b,c)))
        minl = get_smallest(np.transpose((o,a,b,c)))
        # - adjustments
        ax.set_xlim(minl, maxl)
        ax.set_ylim(minl, maxl)
        ax.set_zlim(minl, maxl)
    elif not ax is None:
        fig = ax.get_figure()
    # - lines
    ax.plot(*np.transpose((o, a-o)))
    ax.plot(*np.transpose((o, b-o)))
    ax.plot(*np.transpose((o, c-o)))
    # - labels
    for v, s in zip((a,b,c), (labels)):
        ax.text(*v, s)
    return fig, ax


def plot_line3(ax, v, o=None, label=None) -> None:
    """
    Add a line to an existing matplotlib.axes instance.

    Parameters
    ----------
    ax : matplotlib.axis
        Axis instance to plot on.
    v : np.ndarray
        Cartesian coodinate.
    o : np.ndarray, optional
        Cartesian coordinate. The default is (0,0,0).
    label : str, optional
        Label passed to the line instance. The default is None.

    Returns
    -------
    None
    """
    if o is None:
        o = np.zeros((3,))
    ax.plot(*np.transpose((o, v-o)))
    if label:
        ax.text(*v, label)


def rotation_from_axis_angle(vector:np.ndarray, angle:float, degree:bool=True) -> np.ndarray:
    r"""
    .. math::
        
        R(u, \theta) = \cos(\theta) I + \sin(\theta) u_x + (1 - \cos(\theta)) u \otimes u

    where :math:`u_x` is the cross product matrix.

    Parameters
    ----------
    vector : np.ndarray
        unit vector.
    angle : float
        angle.
    degree : bool, optional
        provided in radians or degrees? The default is degree=True.

    Returns
    -------
    np.ndarray
        Rotation matrix (3,3).

    Reference
    ---------
        `Wikipedia: Rotation_matrix#Rotation_matrix_from_axis_and_angle <https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle>`_
    """
    if degree:
        angle *= np.pi / 180  # as radian
    I = np.eye(len(vector))
    u = vector / LA.norm(vector) # as unit
    uu = np.outer(u,u)
    ux = np.cross(u, -I) # https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
    cos = np.cos(angle)
    sin = np.sin(angle)
    return cos * I + sin * ux + (1 - cos) * uu


def float_tol_pair_in_pairs(pair:np.ndarray, pairs:np.ndarray) -> np.ndarray:
    """
    True if abs(a0 - b0) <= tol & abs(a1 - b1) <= tol for (ai1, aj2), (bi1, bj2)
    in [(a01, a02), ... (aik, ajl)]
    
    NB this is expected to be called in iteration so no sanitization is performed.

    Parameters
    ----------
    pair : np.ndarray
        pair of vectors with shape (2, M)
    pairs : np.ndarray
        collection of vector pairs with shape (N, 2, M)

    Returns
    -------
    np.ndarray
        (pair in pairs) | (pair[::-1] in pairs).
    """
    a = pairs - pair
    # b = pairs - pair[::-1]
    m1 = np.sum( LA.norm(a, axis=2) <= (1e-03, 1e-03), axis=1 ) == 2
    # m2 = np.sum( LA.norm(b, axis=2) <= (1e-03, 1e-03), axis=1 ) == 2
    return m1 #  | m2


# FIXME can accelerate this using sorted input, but not sure how to generalize
def get_unique_pairs(pairs:np.ndarray, mask=False) -> np.ndarray:
    """
    apply float_tol_pair_in_pairs for pair in pairs
    
    Parameters
    ----------
    pairs : np.ndarray
        collection of vector pairs with shape (N, 2, M)
    mask: np.ndarray
        index of unique pairs

    Returns
    -------
    np.ndarray
        (pair in pairs) | (pair[::-1] in pairs) for pair in pairs

    """
    pairs = np.asarray(pairs).reshape((len(pairs), 2, -1))
    rv = [pairs[0]]
    m  = [0,]
    for idx, pair in pairs[1:]: # tqdm(enumerate(pairs[1:]), desc='finding unique pairs...'):
        if not any( float_tol_pair_in_pairs(pair, rv) ):
            rv.append(pair)
            m.append(idx+1)
    if mask is False:
        return np.array(rv)
    return m
