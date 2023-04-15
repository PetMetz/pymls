# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:36:38 2022

@author: pmetz1
"""
# 3rd party
import numpy as np
import pytest

# package
from pymls import toolbox as tbx
from pymls.lattice import Lattice
from pymls.geometric import Dislocation
from pymls.elastic import Stroh, cij_from_group
from pymls.contrast import MLS
# from pymls.contrast import MLS


# --- suites
lattice_suite = \
    ['cubic_lattice',
     'hexagonal_lattice',
     'orthorhombic_lattice',
     'triclinic_lattice'
    ]
    
dislocation_suite = \
    ['cubic_dislocation',
     'hexagonal_dislocation',
     'orthorhombic_dislocation',
     'triclinic_dislocation'
    ]
    
stroh_suite = \
    ['cubic_stroh',
     'hexagonal_stroh',
     'orthorhombic_stroh', 
     'triclinic_stroh'
    ]
    
cij_suite = \
    ['cubic_cij',
     'hexagonal_cij',
     'orthorhombic_cij',
     'triclinic_cij'
    ]

mls_suite = \
    ['cubic_interface',
     'hexagonal_interface',
     'orthorhombic_interface',
     'triclinic_interface'
    ]

# -- cubic crystal systems ----------------------------------------------------
@pytest.fixture
def cubic_lattice():
    """ 3 angstrom cubic lattice """
    return Lattice.from_matrix(3.3065 * np.eye(3))


@pytest.fixture
def cubic_cij():
    """ Im-3m beta Ti-64 constants """
    c11, c12, c44 = 131.4, 98.2, 28.8 # GPa
    return cij_from_group(c11, c12, c44, group='m-3m')


@pytest.fixture
def cubic_stroh(cubic_cij):
    """ Stroh class instance """
    return Stroh(cij=cubic_cij)


@pytest.fixture
def cubic_slip():
    """ (hkl)[uvw] """
    return np.array((1,1,0)), np.array((-1,1,1))


@pytest.fixture
def cubic_dislocation(cubic_slip, cubic_lattice):
    hkl, uvw = cubic_slip
    L = cubic_lattice
    l = np.cross(uvw, hkl) # defines edge dislocation
    phi = tbx.abt(uvw, l, degrees=True) # dislocation character
    return Dislocation(lattice=L, hkl=hkl, uvw=uvw, phi=phi)

@pytest.fixture
def cubic_interface(cubic_dislocation, cubic_cij):
    return MLS(dislocation=cubic_dislocation, cij=cubic_cij) 

# --- hexagonal crystal systems -----------------------------------------------
@pytest.fixture
def hexagonal_lattice():
    """ 3 angstrom cubic lattice """
    return Lattice.from_scalar((3, 3, 4.5, 90, 90, 120))


@pytest.fixture
def hexagonal_cij():
    """ P63/mmc alpha Ti-64 constants """
    c11, c12, c13, c33, c44 = 135.6, 71.1, 24.1, 145.7, 45.7 # GPa
    return cij_from_group(c11, c12, c13, c33, c44, group='6/mmm')


@pytest.fixture
def hexagonal_stroh(hexagonal_cij):
    return Stroh(hexagonal_cij)


@pytest.fixture
def hexagonal_slip():
    """ (hkl)[uvw] """
    return np.array((1,1,0)), np.array((0,0,2))
    

@pytest.fixture
def hexagonal_dislocation(hexagonal_slip, hexagonal_lattice):
    hkl, uvw = hexagonal_slip
    L = hexagonal_lattice
    l = np.cross(uvw, hkl) # defines edge dislocation
    phi = tbx.abt(uvw, l, degrees=True) # dislocation character
    return Dislocation(lattice=L, hkl=hkl, uvw=uvw, phi=phi)

@pytest.fixture
def hexagonal_interface(hexagonal_dislocation, hexagonal_cij):
    return MLS(dislocation=hexagonal_dislocation, cij=hexagonal_cij) 

# --- orthorhombic crystal systems -----------------------------------------------
@pytest.fixture
def orthorhombic_lattice():
    """ Forsterite lattice """
    return Lattice.from_scalar((4.775, 10.190, 5.978, 90, 90, 90))

@pytest.fixture
def orthorhombic_cij():
    """ Forsterite stiffness """
    return cij_from_group(  # GPa
              328.7, # c11
              66.75, # c12
              68.35, # c13
              199.8, # c22
              72.67, # c23
              235.5, # c33
              66.78, # c44
              80.95, # c55
              80.57, # c66
              group='mmm'
              )
        
@pytest.fixture
def orthorhombic_stroh(orthorhombic_cij):
    """ Forsterite dislocation """
    return Stroh(orthorhombic_cij)

@pytest.fixture
def orthorhombic_slip():
    """ (hkl)<uvw> """
    hkl = np.array((0,1,0))
    uvw = np.array((1,0,0))    
    return hkl, uvw

@pytest.fixture
def orthorhombic_dislocation(orthorhombic_slip, orthorhombic_lattice):
    hkl, uvw = orthorhombic_slip
    L = orthorhombic_lattice
    l = np.cross(uvw, hkl) # defines edge dislocation
    phi = tbx.abt(uvw, l, degrees=True) # dislocation character
    return Dislocation(lattice=L, hkl=hkl, uvw=uvw, phi=phi)
    
@pytest.fixture
def orthorhombic_interface(orthorhombic_dislocation, orthorhombic_cij):
    return MLS(dislocation=orthorhombic_dislocation, cij=orthorhombic_cij) 

# --- triclinic crystal systems -----------------------------------------------
@pytest.fixture
def triclinic_lattice():
    """
    Acta Cryst. (1964). 17, 1511
    "The Crystal Structure of Potassium Tetraoxalate, K(HC204)(H2C204). 2H20"
    KH3(C204)2.2 H20
    SpaceGroup: P1 or P-1
    LatticeConstants: (7.04, 10.59, 6.35, 101.4, 100.2, 94.0)
    Density: 1.85 g cm**-3
    Z: 2
    """
    return Lattice.from_scalar((7.04, 10.59, 6.35, 101.4, 100.2, 94.0))
    

@pytest.fixture
def triclinic_cij():
    """
    Acta Cryst. (1970). A26, 401 
    "Triclinic Crystals Ammonium and Potassium Tetroxalate Dihydrate"
    KH3(C204)2.2 H20
    """
    cij = { # x 10^11 dyne cm**-2
           11: 2.536,
           12: 1.184,
           13: 0.983,
           14: 0.072,
           15: 0.612,
           16:-0.123,
           22: 4.779,
           23: 1.402,
           24: 1.134,
           25: 0.146,
           26:-0.270,
           33: 3.430,
           34: 0.219,
           35: 0.147,
           36: 0.040,
           44: 1.019,
           45:-0.082,
           46: 0.053,
           55: 0.569,
           56: 0.070,
           66: 0.499
           } # dicts are ordered in python 3.x
    GPa = 100 * np.array(list(cij.values())) # 10^11 dyne cm**-2 -> 100 GPa
    return cij_from_group(*GPa, group='-1')


@pytest.fixture
def triclinic_stroh(triclinic_cij):
    return Stroh(cij=triclinic_cij)


@pytest.fixture
def triclinic_slip():
    """ (hkl)[uvw] """
    return np.array((0,0,1)), np.array((1,1,1))
    
    
@pytest.fixture
def triclinic_dislocation(triclinic_slip, triclinic_lattice):
    hkl, uvw = triclinic_slip
    L = triclinic_lattice
    l = np.cross(uvw, hkl) # defines edge dislocation
    phi = tbx.abt(uvw, l, degrees=True) # dislocation character
    return Dislocation(lattice=L, hkl=hkl, uvw=uvw, phi=phi)

@pytest.fixture
def triclinic_interface(triclinic_dislocation, triclinic_cij):
    return MLS(dislocation=triclinic_dislocation, cij=triclinic_cij)