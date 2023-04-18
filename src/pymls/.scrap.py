# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 19:23:02 2023

@author: UT
"""


# =============================================================================
#     # FIXME Should be an axis-angle rotation matrix, should confirm again the
#     #       convention of the matrix utilized.
#     @property
#     @tbx.orthogonal
#     @tbx.unit_vectors
#     def Rp2(self):
#         """
#         NB this yields a different rotation matrix than expected from conventional
#         linear algebra, which may again be a convention issue
#         """
#         if self._Rp2 is None:
#             # - (a) e2 := n = Ha* + Kb* + Lc* with coordinates [HKL] in the basis [a*, b*, c*]
#             #       e2 = 1/|n| M @ [h,k,l]
#             # xi2 = self.reciprocal.M.T @ self.hkl / self.reciprocal.length(self.hkl)
#             xi2 = self.xi2
#             # - (b1)
#             phi = np.radians(self.phi)
#             sinp = np.sin(phi)
#             cosp = np.cos(phi)
#             sinp2 = np.sin(phi/2) ** 2
#             xi21, xi22, xi23 = xi2
#             # m1 = 2 * sinp2 * np.array((
#             #     (xi21*xi21,  xi21*xi22, xi21*xi23),
#             #     (xi22*xi21,  xi22*xi22, xi22*xi23), 
#             #     (xi23*xi21,  xi23*xi22, xi23*xi23)
#             #     ))
#             XI2 = xi2 * np.ones((3,3))
#             m1 = 2 * sinp2 * XI2 * XI2.T
#             
#             m2 = np.array((
#                 (1,   xi23, xi22),
#                 (xi23,   1, xi21),     
#                 (xi22, xi21, 1  )
#                 ))
#             m3 = np.array((
#                 ( cosp,  sinp, -sinp),
#                 (-sinp,  cosp,  sinp),
#                 ( sinp, -sinp,  cosp)
#                 ))
#             self._Rp2 =  m1 + (m2 * m3)  # element-wise
#         return self._Rp2
# =============================================================================

# Geometric.P
            # # - (a) e2 := n = Ha* + Kb* + Lc* with coordinates [HKL] in the basis [a*, b*, c*]
            # #       e2 = 1/|n| M @ [h,k,l]
            # modn = self.reciprocal.length(self.hkl)
            # xi2 = 1 / modn * self.reciprocal.M @ self.hkl # np.sqrt( self.hkl @ self.reciprocal.G @ self.hkl)
            # # - (b) e3 := R(phi, e2) @ [b1, b2, b3]
            # modb = self.length(self.uvw)
            # xib = 1 / modb * self.M @ self.uvw #  / np.sqrt(self.uvw @ self.G @ self.uvw)
            # xi3 = self.Rp2 @ xib
            # # - (c) e1 := e2 x e3
            # xi1 = np.cross(xi2, xi3)

    # FIXME These letters from Ting clash with the nomenclature of Stroh
    # --- from Ting ch. 5
# =============================================================================
#     @functools.cached_property
#     def S(self):
#         r"""
#         Barnett-Lothe tensors
#         """
#         return 1j * (2*self.A @ self.B.T - np.eye(3))
#
#     @functools.cached_property
#     def H(self):
#         r"""
#         Barnett-Lothe tensors
#         """
#         return 2j * self.A @ self.A.T
#
#     @functools.cached_property
#     def L(self):
#         r"""
#         Barnett-Lothe tensors
#         """
#         return -2j * self.L @ self.L.T
# =============================================================================
