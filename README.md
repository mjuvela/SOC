## SOC

The directory *SOC* contains the SOC continuum radiative transfer
program implemented with Python and OpenCL. 

The main script for dust emission calculations is ASOC.py. Note that
the map orientations have changed since the previous versions (that
was called SOC.py). The main script for dust scattering calculations
is ASOCS.py. When the program is called, it tries to figure out the
location of the source code and the kernel routines (*.c). These
should always remain in the same directory. ASOC.py and ASOC.py are
development versions (unstable).

make_dust.py is a sample script showing how to convert DustEM files to
the inputs needed by SOC, including the simple combined dust
description (tmp.dust) and the corresponding scattering function file
(tmp.dsc) used by SOC.

To deal with the emission from stochastically heated grains, one uses
"GSET" format dust files written by make_dust.py. In addition,
* A2E_pre.py writes "solver files" for single dust components
* A2E.py solves emission using these "solver files"
* A2E_MABU.py can be used to simplify this when a run includes several 
  dust populations, possibly with spatially varying abundances.
* A2E_LIB.py for making and using the library method (lookup tables
  for faster conversion of absorptions to emission)
* ASOC_driver.py automates the process of (1) calculate absorptions,
  (2) solve dust emission, (3) write emission maps

There are some corresponding julia routines (work in progress). In
particular, DE_to_GSET.jl writes GSET format dust files and the
combined simple ascii file, and the scattering function files needed
by SOC. A2E.jl corresponds to A2E.py and the (still experimental)
script MA2E.jl corresponds to A2E_MABU.py.

Practical examples of the end-to-end calculations will be added here
in the near future. For the moment, one can examine (or even try to
run) the script TEST_LIB.py that includes examples of calculations
with equilibrium temperature dust, with stochastically heated dust,
possibly with spatially varying abundances and possibly sped up by the
use of "library" methods.

For more detailed background and documentation, see 
* http://www.interstellarmedium.org/radiative_transfer/soc/
* Juvela M.: SOC program for dust continuum radiative transfer, 2019,
  A&A 622, A79, https://ui.adsabs.harvard.edu/abs/2019A%26A...622A..79J

