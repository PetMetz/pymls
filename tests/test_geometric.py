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


# --- constants

# --- fixtures

# --- functions
@pytest.mark.parametrize('thisFixture', lattice_suite)
def test_dislocation_instance(thisFixture, request):
    result = request.getfixturevalue(thisFixture)
    assert result


# FIXME this is mostly incomplete and needs to separate tests from expected values

# --- classes
@pytest.mark.parametrize('thisFixture', dislocation_suite)
class TestOrthogonal:
    """ test orthogonal vectors """
    @pytest.fixture(autouse=True)
    def _instantiate_class_fixture(self, thisFixture, request):
        self.d = request.getfixturevalue(thisFixture)

    def test_M(self):
        """ """
        assert tbx.is_orthogonal(LA.inv(self.d.M) * self.d.M)

    def test_G(self):
        """ """
        assert tbx.is_orthogonal(LA.inv(self.d.G) * self.d.G)
    
    def test_reciprocal_M(self):
        """ """
        assert tbx.is_orthogonal(LA.inv(self.d.reciprocal.M) * self.d.reciprocal.M)
    
    def test_reciprocal_G(self):
        """ """
        assert tbx.is_orthogonal(LA.inv(self.d.reciprocal.G) * self.d.reciprocal.G)

    def test_P(self):
        """ """
        assert tbx.is_orthogonal(LA.inv(self.d.P) * self.d.P)
    
    def test_e(self):
        """ """
        assert tbx.is_orthogonal(LA.inv(self.d.e) * self.d.e)
    
    def test_rp2(self):
        """ """
        assert tbx.is_orthogonal(LA.inv(self.d.Rp2) * self.d.Rp2)
    
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

    def test_M(self):
        """ """
        assert self.all_unit(self.d.M)

    def test_G(self):
        """ """
        assert self.all_unit(self.d.G)
    
    def test_reciprocal_M(self):
        """ """
        assert self.all_unit(self.d.reciprocal.M)
    
    def test_reciprocal_G(self):
        """ """
        assert self.all_unit(self.d.reciprocal.G)

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

    def test_M(self):
        """ """
        assert tbx.is_symmetric(self.d.M)

    def test_G(self):
        """ """
        assert tbx.is_symmetric(self.d.G)
    
    def test_reciprocal_M(self):
        """ """
        assert tbx.is_symmetric(self.d.reciprocal.M)
    
    def test_reciprocal_G(self):
        """ """
        assert tbx.is_symmetric(self.d.reciprocal.G)

    def test_P(self):
        """ """
        assert tbx.is_symmetric(self.d.P)
    
    def test_e(self):
        """ """
        assert tbx.is_symmetric(self.d.e)

    def test_rp2(self):
        """ """
        assert tbx.is_symmetric(self.d.Rp2)
    
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
        assert tbx.float_tol(V1, V2)

    def test_reciprocal_construction(self):
        """ """
        V = LA.det(self.d.M)
        x1, x2, x3 = self.d.M
        b1 = np.cross(x2, x3) / V
        b2 = np.cross(x1, x3) / V
        b3 = np.cross(x1, x2) / V
        a1, a2, a3 = self.d.reciprocal.M
        assert tbx.float_tol(a1, b1)
        assert tbx.float_tol(a2, b2)
        assert tbx.float_tol(a3, b3)
        
    def test_reciprocal_metric(self):
        """ will result in same volume but different element-wise """
        A = LA.det( self.d.reciprocal.G )
        B = LA.det( LA.inv(self.d.G) )
        assert tbx.float_tol(A, B)
    # end TestSymmetric