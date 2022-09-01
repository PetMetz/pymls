# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:38:08 2022

@author: pmetz1
"""

import pytest


# --- declare constants
_SMALL = 1e-12
_X1 = 0.000012353487912054


# --- declare fixtures
@pytest.fixture()
def static_quantity():
    return (1, 2, 3)


# --- define test functions
def test_equality():
    """ test for boolean equalities """
    assert bool(1) is True
    
    
def test_error():
    """ test for exceptions """
    with pytest.raises(ZeroDivisionError):
        _ = 1 / 0


def test_with_intermediate_steps(static_quantity):
    """ """
    abc = static_quantity
    s1 = sum(abc)
    s2 = 6
    assert abs(1 - s1 / s2) <= _SMALL
    
    
# --- define test collections
class TestNumbers:
    """ group similar test cases """
    def test_zero(self):
        assert 0 == 0
    
    def test_one(self):
        assert 1 == 1
    
    def test_two(self):
        assert 2 == 2
        
    def test_finite(self):
        assert abs(_X1) >= _SMALL
        

# --- use fixture in class

@pytest.fixture(scope='class')
def other_quantity():
    return (4, 5, 6)

@pytest.mark.usefixtures('other_quantity')
class TestClassMark:
    def test_fixture_equality(self, other_quantity):
        assert other_quantity == (4,5,6)
 

class TestClassFixture:
    
    @pytest.fixture(autouse=True)
    def _var_constructomatic(self, static_quantity):
        self.var = static_quantity
    
    def test_visible(self):
        assert self.var == (1,2,3)
        
        

# --- iterate over multiple inputs
# https://doc.pytest.org/en/latest/proposals/parametrize_with_fixtures.html
# https://miguendes.me/how-to-use-fixtures-as-arguments-in-pytestmarkparametrize
# https://github.com/pytest-dev/pytest/issues/349
# https://stackoverflow.com/questions/42014484/pytest-using-fixtures-as-arguments-in-parametrize

@pytest.fixture
def complex_stuff_1():
    return (1,2,3)

@pytest.fixture
def complex_stuff_2():
    return (4,5,6)

@pytest.fixture
def complex_stuff_3():
    return (7,8,9)

@pytest.mark.parametrize('arg',
                         ['complex_stuff_1',
                          'complex_stuff_2',
                          'complex_stuff_3'])
def test_isreal(arg, request):
    from typing import Iterable
    result = request.getfixturevalue(arg)
    assert isinstance(result, Iterable)
    assert all([not isinstance(e, complex) for e in result])



from typing import Iterable

@pytest.mark.parametrize('thisFixture, thisExpectation', [
                          ('complex_stuff_1', 'True'),
                          ('complex_stuff_2', 'True'),
                          ('complex_stuff_3', 'True')
                          ])
class TestIsReal:
    
    @pytest.fixture(autouse=True)
    def _attr_constructomatic(self, thisFixture, thisExpectation, request):
        self.result = request.getfixturevalue(thisFixture)
        self.expectation = thisExpectation
        
    def test_isreal(self):
        """ not complex is true"""
        assert isinstance(self.result, Iterable)
        assert all([not isinstance(e, complex) is self.expectation for e in self.result])
        
    def test_iscomplex(self):
        """ is complex is not true """
        assert isinstance(self.result, Iterable)
        assert all([    isinstance(e, complex) is not self.expectation for e in self.result])