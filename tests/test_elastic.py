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
from fixtures import orthorhombic_cij, orthorhombic_stroh
from fixtures import triclinic_cij, triclinic_stroh
from fixtures import stroh_suite
from fixtures import cij_suite


# --- constants
O = np.zeros((3,3))
I = np.eye(3)


# --- local fixtures


# --- functions
@pytest.mark.parametrize('thisFixture', cij_suite)
def test_stroh_instance(thisFixture, request):
    """ does it init """
    result = request.getfixturevalue(thisFixture)
    return Stroh(result)


@pytest.mark.parametrize('thisFixture', stroh_suite)
def test_mandel_inversion(thisFixture, request):
    """ CIJ -> cijkl == cijkl' """
    s = request.getfixturevalue(thisFixture)
    Cijkl = np.apply_along_axis(sum, axis=-1, arr=np.transpose(np.indices((3,3,3,3))))
    Cijkl = s.apply_elastic_symmetry(Cijkl)
    Cij = s.apply_mandel(Cijkl)
    A = s.invert_mandel(Cij)
    B = Cijkl
    assert tbx.float_tol(A, B)


@pytest.mark.parametrize('thisFixture', stroh_suite)
def test_voigt_inversion(thisFixture, request):
    """ eij -> ei """
    s = request.getfixturevalue(thisFixture)
    eij = np.apply_along_axis(sum, axis=-1, arr=np.transpose(np.indices((3,3))))
    ei  = s.apply_voigt(eij)
    A = s.invert_voigt(ei)
    B = eij
    assert tbx.float_tol(A, B)


@pytest.mark.parametrize('thisFixture', stroh_suite)
def test_Q(thisFixture, request):
    r""" Q == Q^T`"""
    s = request.getfixturevalue(thisFixture)
    x = np.ones((3,))
    assert tbx.float_tol(s.Q, np.transpose(s.Q))
    assert x @ s.Q @ x > 0 # positve definite


@pytest.mark.parametrize('thisFixture', stroh_suite)
def test_T(thisFixture, request):
    r""" T == T^T`"""
    s = request.getfixturevalue(thisFixture)
    x = np.ones((3,))
    assert tbx.float_tol(s.T, np.transpose(s.T))
    assert x @ s.Q @ x > 0 # positve definite


@pytest.mark.parametrize('thisFixture', stroh_suite)
def test_R(thisFixture, request):
    r""" R != R^T`"""
    s = request.getfixturevalue(thisFixture)
    x = np.ones((3,))
    assert not tbx.float_tol(s.R, np.transpose(s.R))
    assert x @ s.Q @ x > 0 # positve definite


@pytest.mark.parametrize('thisFixture', stroh_suite)
class Test_N1:
    """ """
    @pytest.fixture(autouse=True)
    def _s_class_fixture(self, thisFixture, request):
        self.s = request.getfixturevalue(thisFixture)

    def test_by_construction(self):
        assert tbx.float_tol(self.s.N1, -LA.inv(self.s.T) @ np.transpose(self.s.R) )

    def test_asymmetric(self):
        assert not tbx.float_tol(self.s.N1, np.transpose(self.s.N1))


@pytest.mark.parametrize('thisFixture', stroh_suite)
class Test_N2:
    """ N2 symmetric && N2 positive definite"""
    @pytest.fixture(autouse=True)
    def _s_class_fixture(self, thisFixture, request):
        self.s = request.getfixturevalue(thisFixture)

    def test_by_construction(self):
        assert tbx.float_tol(self.s.N2, LA.inv(self.s.T)) # NB s.T != transpose. I guess this is confusing.

    def test_symmetric(self):
        assert tbx.float_tol(self.s.N2, np.transpose(self.s.N2))

    def test_positive_definite(self):
        x = np.ones((3,))
        assert x.T @ self.s.N2 @ x > 0


@pytest.mark.parametrize('thisFixture', stroh_suite)
class Test_N3:
    """ N3 symmetric && -N3 positive semi-definite """
    @pytest.fixture(autouse=True)
    def _s_class_fixture(self, thisFixture, request):
        self.s = request.getfixturevalue(thisFixture)

    def by_construction(self):
        assert tbx.float_tol(self.s.N3, (self.s.R @ LA.inv(self.s.T) @ np.transpose(self.s.R) - self.s.Q) )

    def test_symmetric(self):
        assert tbx.float_tol(self.s.N3, np.transpose(self.s.N3))

    def test_positive_semidefinite(self):
        x = np.ones((3,))
        assert x.T @ -self.s.N3 @ x >= 0


@pytest.mark.parametrize('thisFixture', stroh_suite)
def test_N_by_definition(thisFixture, request):
    s = request.getfixturevalue(thisFixture)
    # Q1
    assert tbx.float_tol(s.N[:3, :3], s.N1)
    # Q2
    assert tbx.float_tol(s.N[:3, 3:], s.N2)
    # Q3
    assert tbx.float_tol(s.N[3:, :3], s.N3)
    # Q4
    assert tbx.float_tol(s.N[3:, 3:], s.N1.T)


@pytest.mark.parametrize('thisFixture', stroh_suite)
def test_N_not_symmetric(thisFixture, request):
    s = request.getfixturevalue(thisFixture)
    assert not tbx.float_tol(s.N, s.N.T)


@pytest.mark.parametrize('thisFixture', stroh_suite)
def test_eig_by_definition(thisFixture, request):
    s = request.getfixturevalue(thisFixture)
    A = s.N @ s.xi
    B = s.p * s.xi
    assert tbx.complex_tol(A, B)


@pytest.mark.parametrize('thisFixture', stroh_suite)
def test_eig_by_trace(thisFixture, request):
    s = request.getfixturevalue(thisFixture)
    A = np.trace(s.N)
    B = np.sum(s.p)
    assert tbx.complex_tol(A, B)


@pytest.mark.parametrize('thisFixture', stroh_suite)
def test_eig_nontrivial(thisFixture, request):
    r"""
    given :math:`A x = \lambda x` then for non-trivial solutions of x,
    :math:`|A - \lambda I| = 0`.
    """
    s = request.getfixturevalue(thisFixture)
    A = LA.det(s.N - s.p * np.eye(6))
    B = 0j
    assert tbx.complex_tol(A, B)


@pytest.mark.parametrize('thisFixture', stroh_suite)
def test_a_ordering(thisFixture, request):
    s = request.getfixturevalue(thisFixture)
    A = s.a[:, ::2]
    B = np.conjugate(s.a[:, 1::2])
    assert tbx.complex_tol(A, B)
    assert tbx.complex_tol(A, s.A) # by construction


@pytest.mark.parametrize('thisFixture', stroh_suite)
def test_l_ordering(thisFixture, request):
    s = request.getfixturevalue(thisFixture)
    A = s.l[:, ::2]
    B = np.conjugate(s.l[:, 1::2])
    assert tbx.complex_tol(A, B)
    assert tbx.complex_tol(A, s.L) # by construction


@pytest.mark.parametrize('thisFixture', stroh_suite)
def test_p_ordering(thisFixture, request):
    s = request.getfixturevalue(thisFixture)
    A = s.p[::2]
    B = np.conjugate(s.p[1::2])
    assert tbx.complex_tol(A, B)
    assert tbx.complex_tol(A, s.P) # by construction


@pytest.mark.parametrize('thisFixture', stroh_suite)
class TestTingOrthogonalityClosure:
    """ c.f. Ting 5.5 Orthogonality and Closure Relations """

    @pytest.fixture(autouse=True)
    def _s_class_fixture(self, thisFixture, request):
        self.s = request.getfixturevalue(thisFixture)

    # FIXME failing for triclinic fixture (possibly a float tolerance issue)
    def test_Ting_5110(self):
        r"""
        :math:`\left|Q + p(R + R^{T}) + p^2T \right| = 0` (Ting eqn. 5.1-10)
        """
        # arg = lambda s: s.Q + s.P*(s.R + s.R.T) + s.P**2 * s.T
        A = LA.det(self.s.Q + self.s.P*(self.s.R + self.s.R.T) + self.s.P**2 * self.s.T)
        B = 0 + 0j
        assert tbx.complex_tol(A, B)

    # FIXME failing
    def test_Ting_532a(self):
        r"""
        :math:`b = (R^T + pT)a` (c.f. Ting eqn. 5.3-2)
        NB this is the notation of Ting, but given R(3,3) a must be the half-
           set later denoted A
        """
        A = self.s.L  # b
        B = (self.s.R.T + self.s.P * self.s.T) @ self.s.A # (R^T + pT)a
        assert tbx.complex_tol(A, B)

    # FIXME failing
    def test_Ting_532b(self):
        r"""
        :math:`b = (R^T + pT)a = -1/p (Q +pR)a` (c.f. Ting eqn. 5.3-2)
        NB this is the notation of Ting, but given R(3,3) a must be the half-
           set later denoted A
        """
        A = self.s.L  # b
        B = -1 / self.s.P * (self.s.Q + self.s.P * self.s.R) @ self.s.A # (R^T + pT)a
        assert tbx.complex_tol(A, B)

    # FIXME failing
    def test_Ting_551(self):
        """ """
        # x = np.row_stack((self.s.a, self.s.l))
        x = self.s.xi
        p = self.s.p
        L = tbx.square([(-self.s.Q              , O),
                        (-np.transpose(self.s.R), I)
                        ])
        R = tbx.square([(self.s.R, I),
                        (self.s.T, O)
                        ])
        A = L @ x
        B = p * R @ x
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
        r""":math:`N^T (\hat{I} \xi) = p (\hat{I} \xi)` (c.f. Ting eqn. 5.5-8) """
        A = np.transpose(self.s.N) @ (self.s.conI @ self.s.xi)
        B = self.s.p * (self.s.conI @ self.s.xi)
        assert tbx.complex_tol(A, B)

    def test_Ting_559(self):
        r""":math:`\eta = \hat{I} \xi` (c.f. Ting eqn. 5.5-9)"""
        A = self.s.eta
        B = self.s.conI @ self.s.xi
        assert tbx.complex_tol(A, B) # NB this is currently true by definition

    # FIXME failing
    def test_Ting_5510(self):
        r""":math:`\eta_{\alpha} \cdot \xi_{\beta} = \delta_{\alpha \beta}` (c.f. Ting eqn. 5.5-10)"""
        # rv = np.zeros((6,6), dtype=complex)
        # for i in range(6):
        #     for j in range(6):
        #         rv[i,j] = self.s.eta[i] @ self.s.xi[j]
        A = self.s.eta @ self.s.xi
        B = np.eye(6)
        assert tbx.complex_tol(A, B)

    # FIXME failing
    def test_Ting_5513(self):
        r"""
        Orthogonality relations:
        | [BT       AT      ] [A conj(A)] == [I O]
        | [conj(BT) conj(AT)] [B conj(B)]    [O I]
        """
        L = tbx.square([(             self.s.L.T,               self.s.A.T),
                        (np.conjugate(self.s.L.T), np.conjugate(self.s.A.T))
                        ])
        R = tbx.square([(self.s.A, np.conjugate(self.s.A)),
                        (self.s.L, np.conjugate(self.s.L))
                        ])
        A = L @ R
        B = np.eye(6) # tbx.square([(I, O), (O, I)])
        assert tbx.complex_tol(A, B)

    # FIXME failing
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

    # FIXME failing
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

