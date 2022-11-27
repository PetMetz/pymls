# -*- coding: utf-8 -*-
"""

Created on Mon Jun 27 15:18:33 2022

@author: pmetz1
"""

# 3rd party
import numpy as np
from numpy import linalg as LA
import pytest

# package
from pymls.geometric import Dislocation
from pymls import toolbox as tbx

# local
from fixtures import cubic_lattice, hexagonal_lattice, triclinic_lattice
from fixtures import BCC_slip, HCP_slip, triclinic_slip
from fixtures import BCC_dislocation, HCP_dislocation, triclinic_dislocation
from fixtures import lattice_suite
from fixtures import dislocation_suite



# --- patterns ---
'''
@pytest.mark.parameterize('thisFixture', lattice_suite)
def test_():
    ...


@pytest.mark.parametrize('thisFixture', dislocation_suite)
class Test:
    """ ... description ... """
    @pytest.fixture(autouse=True)
    def _instantiate_class_fixture(self, thisFixture, request):
        self.d = request.getfixturevalue(thisFixture)
        
    ...
    
    # end TestOrthogonal
'''

# --- helpers
def MLS_M(lattice):
    """ eqn. 1 """
    D = lattice
    R = lattice.reciprocal
    cos = np.cos(D.angles * np.pi/180)
    sin = np.sin(D.angles * np.pi/180)
    cosstar = np.cos(R.angles * np.pi/180)
    M = np.array((
        (1/D.a                 , 0                 , 0  ),
        (-cos[2] / (D.a*sin[2]), 1 / (D.b * sin[2]), 0  ),
        (R.a * cosstar[1]      , R.b * cosstar[0]  , R.c)
        ))
    return M
    

# --- constants

# --- fixtures

# --- functions
@pytest.mark.parametrize('thisFixture', lattice_suite)
def test_dislocation_instance(thisFixture, request):
    result = request.getfixturevalue(thisFixture)
    assert result


# FIXME this is mostly incomplete and needs to separate tests from expected values

# --- classes

# FIXME (X)^-1 @ X == I is trivially orthogonal. 
@pytest.mark.parametrize('thisFixture', dislocation_suite)
class TestOrthogonal:
    """ test orthogonal vectors """
    @pytest.fixture(autouse=True)
    def _instantiate_class_fixture(self, thisFixture, request):
        self.d = request.getfixturevalue(thisFixture)

    def test_P(self):
        """ """
        assert tbx.is_orthogonal(self.d.P)
    
    def test_e(self):
        """ """
        assert tbx.is_orthogonal(self.d.e)
    
    def test_rp2(self):
        """ """
        assert tbx.is_orthogonal(self.d.Rp2)
    
    # end TestOrthogonal


@pytest.mark.parametrize('thisFixture', dislocation_suite)
class TestAllUnit:
    """ ... description ... """
    @pytest.fixture(autouse=True)
    def _instantiate_class_fixture(self, thisFixture, request):
        self.d = request.getfixturevalue(thisFixture)

    @staticmethod
    def all_unit(x):
        return all( np.apply_along_axis(tbx.is_unit_vector, axis=1, arr=x) )

    def test_P(self):
        """ """
        assert self.all_unit(self.d.P)
    
    def test_e(self):
        """ """
        assert self.all_unit(self.d.e)

    def test_rp2(self):
        """ """
        assert self.all_unit(self.d.Rp2)
    
    # end TestUnit


@pytest.mark.parametrize('thisFixture', dislocation_suite)
class TestSymmetry:
    """ ... description ... """
    @pytest.fixture(autouse=True)
    def _instantiate_class_fixture(self, thisFixture, request):
        self.d = request.getfixturevalue(thisFixture)

    def test_M_asymmetric(self):
        """ """
        assert tbx.is_symmetric(self.d.M) is False

    def test_G_symmetric(self):
        """ """
        assert tbx.is_symmetric(self.d.G) is True
    
    def test_reciprocal_M_asymmetric(self):
        """ """
        assert tbx.is_symmetric(self.d.reciprocal.M) is False
    
    def test_reciprocal_G_symmetric(self):
        """ """
        assert tbx.is_symmetric(self.d.reciprocal.G) is True
    
    # end TestSymmetric
    
    
@pytest.mark.parametrize('thisFixture', dislocation_suite)
class TestComputation:
    """ ... description ... """
    @pytest.fixture(autouse=True)
    def _instantiate_class_fixture(self, thisFixture, request):
        self.d = request.getfixturevalue(thisFixture)

    @staticmethod
    def _vol_from_scalar(a,b,c,al,be,ga):
        cosa = np.cos(al*np.pi/180)
        cosb = np.cos(be*np.pi/180)
        cosg = np.cos(ga*np.pi/180)
        return a*b*c*np.sqrt(1 + 2*cosa*cosb*cosg - cosa**2 - cosb**2 - cosg**2)

    def test_volume(self):
        """ """
        V1 = self._vol_from_scalar(*self.d.scalar)
        V2 = LA.det(self.d.M)
        assert tbx.float_tol(V1, V2) is True

    def test_reciprocal_construction(self):
        """ """
        V = LA.det(self.d.M)
        x1, x2, x3 = self.d.M
        b1 = np.cross(x2, x3) / V  # cyclic permuations
        b3 = np.cross(x1, x2) / V
        b2 = np.cross(x3, x1) / V
        a1, a2, a3 = self.d.reciprocal.M
        assert tbx.float_tol(a1, b1) is True
        assert tbx.float_tol(a2, b2) is True
        assert tbx.float_tol(a3, b3) is True
        
    def test_reciprocal_invertable(self):
        """  """
        A = self.d.reciprocal.G
        B = LA.inv(self.d.G)
        assert tbx.float_tol(A, B) is True
        
    def test_MLS_M_is_transposed(self):
        """ """
        A = MLS_M(self.d)
        B = self.d.M.T
        assert tbx.float_tol(A, B) is True
        
    # end TestSymmetric