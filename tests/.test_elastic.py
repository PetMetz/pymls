# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:19:12 2022

@author: pmetz1
"""

# 3rd party
import numpy as np
import numpy.linalg as LA
import pytest

# package
from pymls import toolbox as tbx
from pymls.elastic import Stroh

# local
from fixtures import cubic_cij, cubic_stroh
from fixtures import hexagonal_cij, hexagonal_stroh
from fixtures import triclinic_cij, triclinic_stroh


# --- constants
O = np.zeros((3,3))
I = np.eye(3)


# --- local fixtures


# --- functions
def test_stroh_instance(cubic_cij):
    """ does it init """
    return Stroh(cubic_cij)


def test_mandel_inversion(cubic_stroh):
    """ CIJ -> cijkl == cijkl' """
    s = cubic_stroh
    Cijkl = np.apply_along_axis(sum, axis=-1, arr=np.transpose(np.indices((3,3,3,3))))
    Cijkl = s.apply_elastic_symmetry(Cijkl)
    Cij = s.apply_mandel(Cijkl)
    A = s.invert_mandel(Cij)
    B = Cijkl
    assert tbx.float_tol(A, B)

    
def test_voigt_inversion(cubic_stroh):
    """ eij -> ei """
    s = cubic_stroh
    eij = np.apply_along_axis(sum, axis=-1, arr=np.transpose(np.indices((3,3))))
    ei  = s.apply_voigt(eij)
    A = s.invert_voigt(ei)
    B = eij
    assert tbx.float_tol(A, B)


def test_Q(cubic_stroh):
    r""" Q == Q^T`"""
    s = cubic_stroh
    x = np.ones((3,))
    assert tbx.float_tol(s.Q, np.transpose(s.Q))
    assert x @ s.Q @ x > 0 # positve definite


def test_T(cubic_stroh):
    r""" T == T^T`"""
    s = cubic_stroh
    x = np.ones((3,))
    assert tbx.float_tol(s.T, np.transpose(s.T))
    assert x @ s.Q @ x > 0 # positve definite
    

def test_R(cubic_stroh):
    r""" R != R^T`"""
    s = cubic_stroh
    x = np.ones((3,))
    assert not tbx.float_tol(s.R, np.transpose(s.R))
    assert x @ s.Q @ x > 0 # positve definite
    

class Test_N1:
    """ """
    @pytest.fixture(autouse=True)
    def _s_class_fixture(self, cubic_stroh):
        self.s = cubic_stroh
        
    def test_by_construction(self):
        assert tbx.float_tol(self.s.N1, -LA.inv(self.s.T) @ np.transpose(self.s.R) )
        
    def test_asymmetric(self):
        assert not tbx.float_tol(self.s.N1, np.transpose(self.s.N1))


class Test_N2:
    """ N2 symmetric && N2 positive definite"""
    @pytest.fixture(autouse=True)
    def _s_class_fixture(self, cubic_stroh):
        self.s = cubic_stroh
        
    def test_by_construction(self):
        assert tbx.float_tol(self.s.N2, LA.inv(self.s.T)) # NB s.T != transpose. I guess this is confusing.
        
    def test_symmetric(self):
        assert tbx.float_tol(self.s.N2, np.transpose(self.s.N2))
        
    def test_positive_definite(self):
        x = np.ones((3,))
        assert x.T @ self.s.N2 @ x > 0
        
        
class Test_N3:
    """ N3 symmetric && -N3 positive semi-definite """
    @pytest.fixture(autouse=True)
    def _s_class_fixture(self, cubic_stroh):
        self.s = cubic_stroh
        
    def by_construction(self):
        assert tbx.float_tol(self.s.N3, (self.s.R @ LA.inv(self.s.T) @ np.transpose(self.s.R) - self.s.Q) )
        
    def test_symmetric(self):
        assert tbx.float_tol(self.s.N3, np.transpose(self.s.N3))
        
    def test_positive_semidefinite(self):
        x = np.ones((3,))
        assert x.T @ -self.s.N3 @ x >= 0
        

def test_N_not_symmetric(cubic_stroh):
    s = cubic_stroh
    assert not tbx.float_tol(s.N, s.N.T)


def test_a_ordering(cubic_stroh):
    s = cubic_stroh
    A = s.a[:, ::2]
    B = np.conjugate(s.a[:, 1::2])
    assert tbx.complex_tol(A, B)
    assert tbx.complex_tol(A, s.A) # by construction


def test_l_ordering(cubic_stroh):
    s = cubic_stroh
    A = s.l[:, ::2]
    B = np.conjugate(s.l[:, 1::2])
    assert tbx.complex_tol(A, B)
    assert tbx.complex_tol(A, s.L) # by construction


def test_p_ordering(cubic_stroh):
    s = cubic_stroh
    A = s.p[::2]
    B = np.conjugate(s.p[1::2])
    assert tbx.complex_tol(A, B)
    assert tbx.complex_tol(A, s.P) # by construction


# --- c.f. Ting 5.5 Orthogonality and Closure Relations
class TestTingOrthogonalityClosure:
    """ c.f. Ting 5.5 Orthogonality and Closure Relations """
    
    @pytest.fixture(autouse=True)
    def _s_class_fixture(self, cubic_stroh):
        self.s = cubic_stroh
        
    def test_Ting_532a(self):
        r"""
        :math:`b = (R^T + pT)a` (c.f. Ting eqn. 5.3-2)
        NB this is the notation of Ting, but given R(3,3) a must be the half-
           set later denoted A
        """
        A = self.s.L  # b
        B = (self.s.R.T + self.s.P * self.s.T) @ self.s.A # (R^T + pT)a
        assert tbx.complex_tol(A, B)

    def test_Ting_532b(self):
        r"""
        :math:`b = (R^T + pT)a = -1/p (Q +pR)a` (c.f. Ting eqn. 5.3-2)
        NB this is the notation of Ting, but given R(3,3) a must be the half-
           set later denoted A
        """
        A = self.s.L  # b
        B = -1 / self.s.P * (self.s.Q + self.s.P * self.s.R) @ self.s.A # (R^T + pT)a
        assert tbx.complex_tol(A, B)
        
    def test_Ting_551(self):
        """ """
        # x = np.row_stack((self.s.a, self.s.l))
        x = self.s.xi
        p = self.s.p
        A = tbx.square([(-self.s.Q              , O),
                        (-np.transpose(self.s.R), I)
                        ])
        A = A @ x
        B = tbx.square([(self.s.R, I),
                        (self.s.T, O)
                        ])
        B = p * B @ x
        assert tbx.complex_tol(A, B)
    
    def test_Ting_552(self):
        """ """
        L = tbx.square([(O,             LA.inv(self.s.T)),
                        (I, -self.s.R @ LA.inv(self.s.T))
                       ])
        R = tbx.square([(self.s.R, I),
                        (self.s.T, O)
                        ])
        A = L @ R
        B = np.eye(6)
        assert tbx.float_tol(A, B)
    
    def test_Ting_553_right_eigenequation(self):
        r"""
        :math:`N \xi = p \xi` (c.f. Ting eqn. 5.5-3) 
        Should be true by construction.
        """
        A = self.s.N @ self.s.xi
        B = self.s.p * self.s.xi
        assert tbx.complex_tol(A, B)
    
    def test_Ting_556_left_eigenequation(self):
        r""":math:`N^T \eta = p \eta` (c.f. Ting eqn. 5.5-6) """
        A = np.transpose(self.s.N) @ self.s.eta
        B = self.s.p * self.s.eta
        assert tbx.complex_tol(A, B)
        
    def test_Ting_558a(self):
        r""":math:`\hat{I} N == (\hat{I} N)^T == N^T \hat{I}` (c.f. Ting eqn. 5.5-8)"""
        A = self.s.conI @ self.s.N
        B = np.transpose(self.s.conI @ self.s.N)
        C = np.transpose(self.s.N) @ self.s.conI
        assert tbx.float_tol(A, B)
        assert tbx.float_tol(A, C)
   
    def test_Ting_558b(self):
        r""":math:`` (c.f. Ting eqn. 5.5-8) """
        A = np.transpose(self.s.N) @ (self.s.conI @ self.s.xi)
        B = self.s.p * (self.s.conI @ self.s.xi)
        assert tbx.complex_tol(A, B)
        
    def test_Ting_559(self):
        r""":math:`\eta = \hat{I} \xi` (c.f. Ting eqn. 5.5-9)"""
        A = self.s.eta
        B = self.s.conI @ self.s.xi
        assert tbx.complex_tol(A, B)
    
    # FIXME  I think these are failing because the solution is semisimple with order 3
    def test_Ting_5510(self):
        r""":math:`\eta_{\alpha} \cdot \xi_{\beta} = \delta_{\alpha \beta}` (c.f. Ting eqn. 5.5-10)"""
        # rv = np.zeros((6,6), dtype=complex)
        # for i in range(6):
        #     for j in range(6):
        #         rv[i,j] = self.s.eta[i] @ self.s.xi[j]
        A = self.s.eta.T @ self.s.xi
        B = np.eye(6)
        assert tbx.complex_tol(A, B)

    def test_Ting_5513(self):
        r"""
        Orthogonality relations:
        | [BT       AT      ] [A conj(A)] == [I O]
        | [conj(BT) conj(AT)] [B conj(B)]    [O I]
        """
        L = tbx.square([(self.s.L.T,               self.s.A.T),
                        (np.conjugate(self.s.L.T), np.conjugate(self.s.A.T))
                        ])
        R = tbx.square([(self.s.A, np.conjugate(self.s.A)),
                        (self.s.L, np.conjugate(self.s.L))
                        ])
        A = L @ R
        B = tbx.square([(I, O), (O, I)])
        assert tbx.complex_tol(A, B)

    def test_Ting_5515(self):
        r"""
        Closure relations (i.e. left and right matrices in eqn. 5.5-13 commute)
        | [A conj(A)]  [BT       AT      ] == [I O]
        | [B conj(B)]  [conj(BT) conj(AT)]    [O I]
        """
        L = tbx.square([(self.s.A, np.conjugate(self.s.A)),
                        (self.s.L, np.conjugate(self.s.L))
                        ])
        R = tbx.square([(self.s.L.T,               self.s.A.T),
                        (np.conjugate(self.s.L.T), np.conjugate(self.s.A.T))
                        ])
        A = L @ R
        B = tbx.square([(I, O), (O, I)])
        assert tbx.complex_tol(A, B)

    def test_Ting_5518(self):
        """ restatement of eigen relations using A, L, P """
        L = tbx.square([(self.s.A, np.conjugate(self.s.A)),
                        (self.s.L, np.conjugate(self.s.L))
                        ])
        P = np.eye(3) * self.s.P
        R = tbx.square([(P, O              ),
                        (O, np.conjugate(P))
                        ])
        A = self.s.N @ L
        B = L @ R
        assert tbx.complex_tol(A, B)

    def test_Ting_5520(self):
        """ diagonalization of N """
        A = self.s.N
        L = tbx.square([(self.s.A, np.conjugate(self.s.A)),
                        (self.s.L, np.conjugate(self.s.L))
                        ])
        P = np.eye(3) * self.s.P
        C = tbx.square([(P, O              ),
                        (O, np.conjugate(P))
                        ])
        R = tbx.square([(self.s.L.T,               self.s.A.T),
                        (np.conjugate(self.s.L.T), np.conjugate(self.s.A.T))
                        ])
        B = L * C * R
        assert tbx.complex_tol(A, B)

    # End TestTing

