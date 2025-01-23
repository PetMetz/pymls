# pymls
Python implementation of the dislocation contrast factor calculation reported by Martinez-Garcia, Leoni, and Scardi. [^1]

## About
For example calculations, see `Examples`.
To run existing tests, execute `tests/run_tests.bat` or individual `tests/test_*.py`.
This is mostly linear algebra, so it's pretty fast. For some slopy profiling, run your favorite profiler on `tests/profile_MLS.py`.

## Problems
Well, just look at the test coverage. These are largely based on the discussion of the Stroh method given by Ting (ch. 5).[^2]
So far, values computed for titanium disagree with ANIZC.[^3]
Further, values computed for Forsterite (a-Mg2SiO4) disagree with those reported in Martinez-Garcia et al.[^1]
The algebra is failing at some point I haven't understood.

## Installation
0. create a branch if desired
1. clone repo to local drive
2. navigate to the directory containing `~/pymls`
3. in command prompt, execute `pip install -e pymls`
4. I haven't tested to see if the installation will work on linux (it should)

## Dependencies
- Python 3+
- numpy, scipy, matplotlib

## References
[^1]: Martinez-Garcia, J., Leoni, M., & Scardi, P. (2009). A general approach for determining the diffraction contrast factor of straight-line dislocations. Acta Crystallographica Section A Foundations of Crystallography, 65(2), 109–119. https://doi.org/10.1107/S010876730804186X
[^2]: Ting, T. T. C. (1996). The Stroh Formalism. In Anisotropic Elasticity. Oxford University Press. https://doi.org/10.1093/oso/9780195074475.003.0008
[^3]: Borbély, A., Dragomir-Cernatescu, J., Ribárik, G., & Ungár, T. (2003). Computer program ANIZC for the calculation of diffraction contrast factors of dislocations in elastically anisotropic cubic, hexagonal and trigonal crystals. Journal of Applied Crystallography, 36(1), 160–162. https://doi.org/10.1107/S0021889802021581
