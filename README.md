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

For more detailed background and documentation, see 
* http://www.interstellarmedium.org/radiative_transfer/soc/
* Juvela M.: SOC program for dust continuum radiative transfer, 2019,
  A&A 622, A79, https://ui.adsabs.harvard.edu/abs/2019A%26A...622A..79J

## Input Files

### Keywords (alphabetical order)

| Keyword | Type | Default | Description |
|---|---|---|---|
| `3d_map` | nx ny nz | — | 3D map dimensions. |
| `absthin` | int | — | Absorption thinning. |
| `absorbed` | filename | — | Gives the name for the file that will contain the number of absorbed photons in each cell and frequency. |
| `ali` | int | — | Turns on accelerated lambda iterations (only for dust emission, in calculations with multiple iterations, not for stochastically heated grains). |
| `background` | filename [k] | — | Specifies the isotropic background where the intensities are read from file filename, optionally scaled by a factor of k. |
| `bgmethod` | int | — | Version of the BGPAC simulation part. |
| `bgpackets` | int | — | Specifies the number of photon packages to be simulated (per iteration and per frequency) from the background (see `background` and `hpbg`). |
| `batch` | int | — | Sub-batch size of rays per kernel call. |
| `cellpackets` | int | — | Specifies the number of photon packages to be simulated (per iteration and per frequency) from dust within the model volume. |
| `CLE` | int | — | Tells SOC to calculate dust emission on device instead of in the host Python code (much faster; only for equilibrium dust). |
| `CLT` | int | — | Tells SOC to calculate dust temperatures on device instead of in the host Python code (much faster; only for equilibrium dust). |
| `cload` | filename | — | Tells SOC to read absorptions from the specified file that has been written in a previous SOC run using the `csave` option. In this case the simulation of "constant" radiation sources (everything apart from the dust medium itself) will be omitted. Initial run with `csave` can be combined with subsequent SOC runs with `cload` to avoid the unnecessary overhead of resimulating the absorptions caused by those constant radiation sources. |
| `cloud` | filename | — | Gives the name of the cloud file (containing densities and information of the grid hierarchy). |
| `colden` | filename | — | Asks for column density maps (corresponding to the actual maps being calculated) to be saved to file named based on filename – the keyword `colden` has been replaced with `savetau`. |
| `csave` | filename | — | Tells SOC to save absorptions to the given file; the file will contain the absorptions that result from all radiation sources except for the dust medium itself; the file can be read in subsequent SOC runs by using the `cload` keyword. |
| `CR_HEATING` | float | — | Cosmic ray heating multiplier (0.0 = no CR heating). |
| `DEFS` | string | — | To ease experimentation with different kernels, this can be used to add definitions that are passed to the kernel at compile time; this is a string that should follow the format "-D definition" (or any strings understood by the compiler). |
| `density` | float | 1.0 | Specifies additional scaling for the density values that are read from the cloud file (default is 1.0, no rescaling). |
| `device` | string [#] | — | Specifies the device (in some versions devices) to be used; c stands for CPU (the default) and g for GPU. The string could thus be 'c' or 'g', or 'cg' (to allow either CPU or GPU to be chosen). One can also give a second optional parameter, which chooses a particular device based on string in its name. For example "device c Intel" or "device g GTX". The alternative way to choose a particular device is to use the `platform` keyword. |
| `direwei` | float, float | — | Direction weighting parameters. |
| `diffpack` | int | — | Gives the number of photon packages (per frequency) that are used to simulate emission from the diffuse emission sources specified with `diffuse`; this is separate from the dust emission from the medium. |
| `diffuse` | filename [#] | — | Specifies a file with additional sources of diffuse radiation (photons/cm³/Hz); this is a plain binary file containing CELLS and NFREQ (two int32 numbers) followed by CELLS float32 numbers for the actual emission; see keyword `diffpack`; note: this diffuse emission is completely separate from the emission that is calculated for the actual dust medium. |
| `directions` | float, float | — | Specifies the direction towards the observer that is used when calculating the output maps; theta is the angle from positive Z axis and phi the rotation around the +Z axis, the angle increasing from +X towards +Y. For direction 90 0, the observer is in the direction of the positive X axis and the maps are oriented so that cloud Y axis is right and the Z axis up. For direction 0 0, observer is in the direction of positive Z axis and in the maps Y is right and X is down. For direction 90 90, the observer is in the direction of the positive Y axis and in the maps the cloud X axis points left and the cloud Z axis up. |
| `distance` | float | — | Distance of the model [pc]. |
| `dsc` | filename | — | Specifies the name for the scattering function file; there should be either a single line with `dsc` or as many instances as there are dust components. |
| `dustfile` | filename | — | Optical cross section for polarisation. |
| `emitted` | filename | — | Specifies the name for a file where the emissions (number of emitted photons per cell and frequency) are stored. |
| `emweight` | int, int, int, int | — | For numbers >0, generates photon packages from the dust medium according to a weighting scheme (currently options emweight 1 and emweight 2 exist). Additional optional arguments: minimum packets per cell, maximum packets per cell, threshold to ignore cells. |
| `EXEC` | string | — | Executes the remainder of the line in a shell at the time when the ini-file is read. Because of obvious security concerns, this is by default commented out in the code. |
| `externalm` | filename | — | Mask for emitting cells (value 1 = emit, 0 = do not emit). |
| `fast_map` | int | — | Fast map calculation (>1 = several frequencies in parallel). |
| `fits` | float, float | — | Save spectra to FITS file instead of a plain binary file. Ra and de are optional image centre coordinates [degrees] used in the file header. |
| `ffs` | int | 1 | Turns on (#=1) or off (#=0) the method of forced first scattering. This applies only to the SOCS.py script where the forced first scattering method is by default on. |
| `global` | int | — | Overrides the GLOBAL work group size in ASOC.py. |
| `gridlength` | float | — | The size of a root grid cell in parsecs. |
| `hpbg` | filename, float, int | — | Specifies a Healpix file that is used to describe the surface brightness of the background sky (instead of or in addition to an isotropic background); the first optional numerical parameter is a scaling factor applied to the intensity values in the file (default=1) and the second optional parameter, if >0, tells SOC to use weighted sampling when generating photon packages from the background. |
| `interpol` | int | — | If healpix map-making uses interpolation. |
| `iterations` | int | — | Number of iterations (when dust emission is simulated and the self-coupling between this emission and dust temperature is significant). |
| `loadtemp` | filename | — | In case of equilibrium dust (and only a single dust population), recalculates emission based on dust temperatures in the given file; this is used mainly to save disk space, storing CELLS temperature values instead of directly the CELLS×NFREQ emission values. |
| `local` | int | — | Overrides the default size of OpenCL work groups; the defaults are 8 for CPU and 32 for GPU. |
| `mapcent` | float, float, float | — | Position of the centre of the maps. |
| `mapping` | int, int, float | — | Specifies the number of pixels and the pixel size of the output maps; nx and ny are the map dimensions and step the pixel size in units of the root grid cells (=in units of gridlength). |
| `mapint` | int | — | If normal mapmaking (not healpix) uses interpolation. |
| `mapum` | float, float, ... | — | List wavelengths [µm] for which (and only for which) maps should be written; this is used together with the FITS keyword, to produce separate FITS files for the surface brightness at the listed wavelengths. |
| `mirror` | string | — | Changes simulation so that photon packages that would exit the model volume are reflected back on selected boundaries; in the argument string (1-6 characters long), lower-case letters refer to lower boundaries and upper-case letters to upper boundaries along the corresponding coordinate axes. |
| `mmapabs` | int | 1 | Selects the absorption data to be handled using a normal file (I=0) or as a memory mapped file (I=1, default). Option I=1 requires less memory (one frequency at a time) but option I=0 is significantly faster, if data for all frequencies can be kept in the main memory during the run. |
| `mmapemit` | int | 1 | Selects the emission data to be handled using a normal file (I=0) or as a memory mapped file (I=1, default). Option I=1 requires less memory but option I=0 is faster, if data for all frequencies can be kept in the main memory. |
| `nnabs` | float, float, ... | — | For runs involving neural networks, the list of wavelengths at which the intensities are used as the inputs for the networks. |
| `nnemit` | float, float, ... | — | For runs involving neural networks, the list of wavelengths for which the networks provides intensity predictions. |
| `nnmake` | string | — | Instructs SOC to create a mapping from absorptions (at wavelengths specified by `nnabs`) to emission (at wavelengths specified by `nnemit`). The trained networks are saved to files consisting of the string provided by the keyword `nnmake` and the name of the dust component. |
| `nnnlimit` | int | — | Density threshold for NN training sample. |
| `nnthin` | int | — | Instructs SOC to solve the emission only for every #:th cell, to be used in the training of neural networks (with keyword `nnmake`). |
| `nnsolve` | string | — | Instructs SOC to use neural networks to predict the emission (at wavelengths specified by `nnemit`), based on RT simulations at wavelengths specified by `nnabs`. |
| `noabsorbed` | — | — | In dust emission calculations with equilibrium-temperature dust (no stochastically heated grains), the integral of absorbed energy in each cell can be calculated on the fly, without saving the per-frequency values to a file; by using this keyword, one can in that case skip the writing of the large (CELLS×NFREQ×4 bytes) file of absorptions. |
| `nomap` | — | — | Skip the writing of intensity and polarisation maps. |
| `nosolve` | — | — | Skip the solving of dust temperatures and dust emission inside SOC (even when calculating a single dust species with grains in equilibrium temperature; used for example when SOC should write spectrum files based on emission that was already calculated in a previous run or with an external program). |
| `optical` | filename [filename] | — | This specifies the name of the file containing the dust optical parameters (see Dust Files below); the second optional parameters gives the name of a binary file of [CELLS] float32 numbers that specifies the relative abundance of the dust in every cell (the default would be 1.0 for every cell). |
| `optishalf` | — | — | When calculations include multiple dust components with variable abundances, we need to pass to the kernel large arrays that contain absorption and scattering cross section separately for each cell – with this keyword these are passed as half-precision floating point numbers (16 bits) allowing larger models to be calculated (especially on GPUs) but with some loss of accuracy (saving 4×CELLS bytes of device memory). |
| `outnside` | int | — | NSIDE for scattering maps (healpix). |
| `perspec` | float, float, float | — | Location of an internal observer inside the model (root coordinates). |
| `platform` | int, int | 0, 0 | Select a specific OpenCL platform; by default SOC uses the first available OpenCL platform, based on the CPU vs. GPU selection made with the keyword `device`. One can also give an optional second integer argument, which chooses a particular device from the given platform. The default is "platform 0 0", while "platform 1 2" would choose the second platform and its third device (e.g. if one had multiple GPUs). |
| `p0` | float | — | If keyword `polred` is not used, specifies the intrinsic polarisation fraction (= polarisation fraction for uniform plane-of-the-sky magnetic field); note that the keyword is letter "p" followed by zero (0, not o). |
| `pointsource` | float, float, float, filename, float | — | Adds a point-like radiation source to the model; the position is given by (x,y,z) in units of the root grid, filename refers to the luminosity file (see section Radiation Field below), and scale is an optional scaling (a floating point number) for the source luminosity; there can be many lines with this keyword. |
| `polmap` | filename, filename, filename, int | — | Asks SOC to write out (I, Q, U) maps using the three data files given as the arguments; these are hierarchical files with the same structure as the density file and contain the components of the magnetic field vectors for each cell. Optional arguments: MAXLOS, MINLOS, MAXLOS (maximum/minimum length of the line of sight). |
| `polrhoweight` | — | — | Polarization rho weighting (density-weighted instead of emission-weighted). |
| `polred` | filename | — | File of polarisation reduction factors R for each cell; the format is a plain binary file with the number of cells (int32) followed by CELLS floating point numbers (float32); the input file format could be changed in the code, also to a hierarchical file with the same format as the density file. |
| `polstat` | int | — | Write maps of polarisation statistics: S, <cos_theta>, ... |
| `prefix` | string | — | Specifies the prefix attached to the names of some output files. |
| `psmethod` | int | — | Alternative schemes for the simulation of point sources when they are located outside the model volume; the default is to simulate photon packages isotropically; method 1 sends packages only in the 2π solid angle that contains the cloud (a small improvement in SNR); method 2 sends photon packages only towards visible cloud surface elements but the weighting between visible cloud sides is not quite optimal; method 5 sends photon packages towards a cone that is aligned with one of the coordinate axes and contains the cloud (less optimal if the point source position is far from the cloud centre when both are projected onto the plane perpendicular to the centre direction of the cone); currently `psmethod` 2 is still the recommended one, especially if the point sources is far outside the model volume. |
| `pspackets` | int | — | Specifies the number of photon packages (per frequency and iteration) that are used to simulate emission from point sources. |
| `pssavetau` | filename, float | — | File to save column density and optical depth towards point sources; float is the wavelength in µm. |
| `radiusalign` | filename | — | Alignment radius data. |
| `reference` | — | — | Turns on the use of a reference field (only for dust emission with multiple iterations, in practice not for stochastically heated grains). |
| `remit` | float, float | — | Limits the produced emission file (see keyword `emitted`) to contain data only for the given wavelength range, specified with lower and upper limits in micrometers; used mainly to reduce the size of the output files when surface brightness data are needed only for a limited set of wavelengths (note: in case of stochastically heated grains, the helper script A2E_MABU.py will also read this option). |
| `savetau` | filename, float | — | Save map of optical depth to the named file for the wavelength specified in units of µm. If the wavelength is omitted, a map of the column density is saved instead. The map (pixel size, number of pixels) is the same as for regular maps (see `mapping`). |
| `saveint` | int | — | Saves intensity values for all cells; if parameter # is equal to 1, SOC saves to ISRF.DAT intensities estimated from photons entering a cell (two int32 numbers CELLS, NFREQ followed by float32 array [CELLS, NFREQ]); with value 2 it saves to the file also the net vector sum using additional three vector components (file begins with three int32 numbers CELLS, NFREQ, 4 that are followed by the float32 array [CELLS, NFREQ, 4]); with parameter value 3 SOC saves intensities that are calculated based on the absorptions within the cell (effective value for the intensity over the cell volume, not the average intensity on the cell boundary), the file starts with two int32 numbers (CELLS, NFREQ) followed by the float32 array [NFREQ, CELLS] of intensity values in cgs units but scaled with 4π – i.e. values are 4×π×I, not directly intensity I. |
| `seed` | float | — | Specifies a floating point seed value for the random number generation; if the provided value is negative, the seed value is taken from the clock. |
| `singleabu` | — | — | In the special case of calculation having two dust components for which the sum of relative abundances is one, this advises SOC to store internally only an array for the abundances of the first dust component, the second having implicitly abundances one minus the abundance of the first dust (saving 4×CELLS bytes of host memory). |
| `simum` | float, float | — | Specifies in units of µm the wavelength range used in the simulations (can be used for example to skip long wavelengths that do not contribute to heating); default range is 0.003 µm–3 m. |
| `solveondev` | — | — | If >0, solve temperatures on device. |
| `sourcemap` | filename | — | File to save map of attenuated sources. |
| `stepwei` | int, float, float | — | Step-length weighting parameters: (0) unweighted, (1) single exponential with multiplier, (2) two exponentials with weight and multiplier arguments. |
| `temperature` | filename | — | In case of calculations with a single equilibrium temperature dust, specifies a file where the dust temperatures of all cells are stored (the hierarchical file format is the same as for the cloud density file). |
| `threshold` | int | — | If >0, save (polarisation) maps ignoring large cells below this hierarchy level. |
| `verbose` | int | — | Output verbosity level. |
| `wavelengths` | float, float | — | Specifies an optional smaller wavelength range for the output spectrum files (resulting in smaller files, if spectra are needed only for a smaller wavelength range); the arguments are the minimum and maximum wavelength in units of µm. |
| `yshear` | int | — | For shearing box simulation, implies periodicity in xy plane. |

### Additional Keywords (Not in Online Documentation)

| Keyword | Type | Default | Description |
|---|---|---|---|
| `3d_map` | int, int, int | — | 3D map dimensions. |
| `absthin` | int | — | Absorption thinning. |
| `bgmethod` | int | — | Version of the BGPAC simulation part. |
| `batch` | int | — | Sub-batch size of rays per kernel call. |
| `CR_HEATING` | float | — | Cosmic ray heating multiplier (0.0 = no CR heating). |
| `direwei` | float, float | — | Direction weighting parameters. |
| `distance` | float | — | Distance of the model [pc]. |
| `dustfile` | filename | — | Optical cross section for polarisation. |
| `externalm` | filename | — | Mask for emitting cells (value 1 = emit, 0 = do not emit). |
| `fast_map` | int | — | Fast map calculation (>1 = several frequencies in parallel). |
| `global` | int | — | Overrides the GLOBAL work group size in ASOC.py. |
| `interpol` | int | — | If healpix map-making uses interpolation. |
| `mapcent` | float, float, float | — | Position of the centre of the maps. |
| `mapint` | int | — | If normal mapmaking (not healpix) uses interpolation. |
| `nnnlimit` | int | — | Density threshold for NN training sample. |
| `outnside` | int | — | NSIDE for scattering maps (healpix). |
| `perspec` | float, float, float | — | Location of an internal observer inside the model (root coordinates). |
| `polrhoweight` | — | — | Polarization rho weighting (density-weighted instead of emission-weighted). |
| `polstat` | int | — | Write maps of polarisation statistics: S, <cos_theta>, ... |
| `pssavetau` | filename, float | — | File to save column density and optical depth towards point sources; float is the wavelength in µm. |
| `radiusalign` | filename | — | Alignment radius data. |
| `sourcemap` | filename | — | File to save map of attenuated sources. |
| `stepwei` | int, float, float | — | Step-length weighting parameters: (0) unweighted, (1) single exponential with multiplier, (2) two exponentials with weight and multiplier arguments. |
| `threshold` | int | — | If >0, save (polarisation) maps ignoring large cells below this hierarchy level. |
| `verbose` | int | — | Output verbosity level. |
| `yshear` | int | — | For shearing box simulation, implies periodicity in xy plane. |

### Cloud Description

We describe here the format for the density file that uses a hierarchical grid. This is a binary file where all values are either 4 byte integers or 4 byte floats. The file begins with five integers: [Nx, Ny, Nz, levels, cells]. Here Nx-Nz are the dimensions of the cartesian root grid, levels is the number of hierarchy levels and cells the total number of cells in the model. This is followed, for each level in the grid hierarchy, by the number of cells on the current level and the vector of density values for the cells on that level. The file thus in a sense consists of separate density vectors, one for each hierarchy level of the discretisation. We follow the C-order where the index of the X-axis is running fastest.

If the whole hierarchy only contains the root grid, the file thus starts with the numbers [Nx, Ny, Nz, 1, NX×Ny×Nz, NX×Ny×Nz], where 1 tells that there is only one level. In this case the number NX×Ny×Nz appears twice, first as the total number of cells, then as the number of cells on the first level. These six integers are followed by NX×Ny×Nz floating point numbers. Note that if you write the density cube from a 3D python array, the last index is running fastest so that for SOC the Python array element n[k,j,i] would correspond to coordinates x=i, y=j, and z=k.

If the cloud contains multiple hierarchy levels, the density value of a refined cell should be replaced with a link to its first subcell that is part of the density vector for the next hierarchy level. If the first subcell is the k:th cell in the vector containing all cells of the next level, then the density value of the parent cell is replaced by the bit representation of the integer number -k. The fact that it is negative tells SOC that it cannot be a density value and it is thus interpreted this way as a way to find the child cells.

Note that in bit representation the value is still integer, not a floating point value like the rest of the elements in the density vector. Thus, when working with Python, one cannot simply take a negative value m from the density vector of floats, multiply it by -1 and assume that int(-m) is the index of the child cell. You have to convert it to integer bit representation using Python ctypes library, something like `ctypes.c_int.from_buffer(ctypes.c_float(m)).value`. The converse transformation from integer to floating point number would be `ctypes.c_float.from_buffer(ctypes.c_int(i)).value`.

The density values are typically density of hydrogen atoms per cm³ or density of hydrogen molecules per cm³. The correct unit is determined by the dust model, relative to which density the dust opacities are defined.

### Dust Description

#### Standard file format - the "simple" format

SOC concentrates on the actual radiative transfer (RT) simulations and was written to be dust model agnostic. Many things that are later needed to compute dust temperatures are not needed during the RT phase, even when model includes several possibly stochastically heated dust populations. Therefore the input file for dust properties can be quite simple and is made only slightly more complex by being backwards compatible with the format used for the CRT program.

We refer to this file format as the "simple" format. Here is an example of the beginning of the file that in the SOC ini file is specified using the optical keyword.

```
eqdust
1.00000e-07      # number density of grains
1.00000e-04      # grain size [cm]
184              # number of frequencies
# freq        g         Qabs         Qsca
2.99792e+09   0.00000   6.22363e-15  2.97781e-27
3.67645e+09   0.00000   7.83909e-15  6.73353e-27
...
```

The first line is always eqdust (format assuming that grains are at some equilibrium temperature, ignoring parameters needed to solve temperature distributions of stochastically heated grains). The second line contains the ratio x between the number of grains and the number of particles, whose number density is given in the cloud file. The third row is the radius a of the grains [cm] and the forth row the number of frequencies in the main table. The main table below the four-line header has four columns: frequency, asymmetry parameter g of the scattering function, absorption efficiency Qabs and scattering efficiency Qsca. The actual cross section for absorption per unit density is thus pi×a²×Qabs×x and similarly the cross section for scattering per unit density is pi×a²×Qsca×x and only the values of these products are used.

In the ini-file keyword `dsc` is used to specify the file that contains data on the discrete scattering function. The file contains two data tables. The first one is discretised linearly as a function of the cosine of the scattering angle theta (corresponding to linspace(-1.0,+1.0, bins) where bins is the number of bins). For each value of cos(theta), one lists the scattering probability per unit solid angle (not per unit step in theta or in cosine of theta!). This is repeated for every frequency used in the simulation. The second half of the file includes the cumulative probability functions of cos(theta) that provide a mapping from values [0,1] to [-1,+1]. SOC uses this directly to generate random scattering directions. All in all, the scattering function file consists of bins×NFREQ+bins×NFREQ floats of 4 bytes each.

When runs include several dust species with varying abundances, SOC must have information of the dust opacities and scattering properties separately for each dust component. In that case the ini-file should just have one line with `optical` and one line with `dsc` keyword per each dust component (of course also specifying dust-specific abundances). If one has many dust components but their relative abundances do not vary over the model volume, it is better (=faster) to combine them into a single dust species with a single scattering function.

#### File format for stochastically heated grains

Although SOC itself has no knowledge of stochastic heating, one can use it for calculations of stochastically heated grains. In that connection other programs use a more complete file format for the dust populations, including the information of the grain size distributions and grain enthalpies. For historical reasons, this is called the "gs" format.

Here is an example of a "gs" file for astrosilicates:

```
gsetdust
prefix     aSilx
nstoch     -1
optical    gs_aSilx.opt
enthalpies gs_aSilx.ent
sizes      gs_aSilx.size
```

The file starts with the line gsetdust, stating that the file conforms to the "gs" format. Otherwise the main content consists of references to files for optical properties, enthalpies and sizes.

The file of optical properties begins with NSIZE and NFREQ, followed by the grain size in micrometers, then a table (freq, Qabs, Qsca, g) for each size. The enthalpy file contains size and temperature grids followed by enthalpy values (per grain, in cgs units). The size distribution file contains GRAIN_DENSITY, NSIZE, NE, and for each size bin: SIZE [um], S_FRAC, Tmin [K], Tmax [K].

### Radiation Field Prescription

Radiation field can be produced by isotropic and non-isotropic background sources and point-like radiation sources that reside inside or outside the actual model volume.

**Isotropic background** — Specification of the intensity of an isotropic background component requires a binary file that lists the intensity in cgs units for each frequency (in the order of increasing frequency) included in the simulation. The frequency grid must be identical to the one specified by the keyword `optical`. It is thus a plain binary file with NFREQ 4 byte floating point numbers.

**Anisotropic background** — Anisotropic background is specified using a Healpix-type binary file. At the moment the resolution of the background map is hardcoded to correspond to healpix maps with size NSIDE=64 (49152 pixels over the sky, each pixel ~55 arcmin in size) and in the RING order. The file simply lists the pixel values (4 byte floats) for 49152 pixels on the first frequency, followed by similar vectors for the remaining frequencies in the order of increasing frequency. The total size of the file is thus always NFREQ×49152×4 bytes.

**Point sources** — For point sources one of course specifies luminosity (in cgs units, erg/s/Hz) instead of intensity. The file is a plain binary file with NFREQ 4 byte floats, one number per simulated frequency, in the order of increasing frequency. The frequency grid must again be identical to the one specified by the `optical` keyword.

## Output Files

### Dust Emission

SOC.py (and the development version ASOC.py) saves the dust emission maps to binary files `map_dir_#.bin` where # is a running index of the observer direction, corresponding to the inifile `directions` keyword. Each file begins with the map dimensions NX and NY in pixels (2×int32), followed by the data cube that in Python corresponds to an array with dimensions (NFREQ, NY, NX). Here NFREQ is the number of frequencies that usually is the same as specified by the keyword `optical` but may be limited by the keyword `wavelengths`.

If the output maps are calculated in the healpix format, the files have names `map_dir_0_H.bin`. The file begins with four int32 numbers: NSIDE, -1, NFREQ, LEVELS. NSIDE is the NSIDE parameter of the Healpix maps, the second number is a dummy negative value, NFREQ is the number of frequencies and levels is the number of levels in the hierarchical discretisation of the model cloud. The rest of the file contains the allsky healpix maps that as a Python array corresponds to an array (NFREQ, 12×nside×nside) of float32 numbers.

### Scattered Light

The scattered light images computed with SOCS.py (and ASOCS.py) are stored in binary files called `outcoming.dat` that also contain the different observer directions within the same file. The file begins with the number of pixels in the y and x directions and the number of frequencies (3 × int32). These are followed by the list of frequencies (float32 values). The rest of the file forms a data cube (float32 values) that in Python has dimensions (nfreq, ndir, ny, nx): the number of frequencies, the number of observer directions, and the dimensions of each scattered radiation map.

## Basic SOC Runs

In this section we look at basic SOC runs that involve a model with Cartesian spatial discretisation and a single dust population, calculations ignoring the stochastic heating of the dust grains. This would be appropriate mainly for the FIR and longer wavelengths >100 µm - with the caveat that the long-wavelength emission will be slightly overestimated because stochastic heating would move some of that radiated energy to shorter wavelengths.

When there is only a single dust component and stochastic heating is not taken into account, the dust temperatures and dust emission can be calculated inside SOC. Thus, a single SOC run will do all three steps: simulate the radiation field, recompute dust temperatures and emission, and write the results.

To optimise the actual runs, it is possible to split them into smaller steps. With the ini file keywords `nosolve` and `nomap` one can first do radiation field simulation only. Even further, one can separate the simulations of the dust reemission from the simulations of other (immutable) radiation sources. For example, one could first simulate radiation from the diffuse background and point sources, and save the result with the `csave` keyword. If one needs to iterate the calculations to find the dust equilibrium temperatures (and one does not know beforehand how many iterations might be needed), one can then remove the `nosolve` keyword, add the `cload` keyword to read in the precomputed part of the radiation field. Once the dust temperatures have converged, a separate SOC run can be done to write the spectra (a run with zero iterations, after removing the `nomap` keyword). When there is only a single, equilibrium-temperature dust component (i.e. no stochastic heating), one can safely use the keyword `noabsorbed` to save some disk space.

## Advanced SOC Runs

### Hierarchical Discretisation

The use of hierarchical discretisation in principle only means that one adds more hierarchy levels in the cloud file. Otherwise the main consideration is the number of photon packages in the simulation. This affects especially the simulation of the background radiation entering the model volume. Each additional level of refinement means that the smallest cells have a factor of four smaller projected area. If the number of background photon packages (keyword `bgpackets`) is kept constant, the number of rays hitting those smallest cells goes by a factor of four and the noise in the dust temperature estimates in principle doubles. With deeper hierarchies, one will also need many more photon packages. One can also use the keyword `split`, which means that simulated photon packages are split into four each time the refinement level increases along the photon path.

Another affected parameter is the number of pixels in the surface brightness maps. Even if the maps are finally needed at lower resolution, there is little use to make a model with many hierarchy levels unless the emitted radiation is sampled at a corresponding resolution. However, while the cloud model uses octree discretisation to reduce the amount of memory used, the output files use a fixed pixel size. When the map pixel size is set to match the smallest model cell size, the map files can become large. For example, for a model with a root grid of 512×512×512 cells and six levels of refinement, a single map would be 4 GB - potentially more than the whole 3D model cloud file itself. It is advisable to limit the number of frequencies in the map files to the absolute minimum using keywords `wavelengths` or `libmap`.

### Multiple Dust Components

When there are multiple dust populations with spatially varying abundances, dust emission is no longer computed inside the SOC program itself. SOC is still used to (1) simulate the radiation field and (2) to write the dust emission spectra. However, between these two steps one needs to run another script `A2E_MABU.py`, which reads the radiation field information written by SOC, splits the number of absorptions between the dust species, solves the temperatures and emission for each dust species separately, and finally adds the results into a single file describing the total dust emission (in each cell and at each frequency).

The steps are:

1. **Initial radiation field simulation** — After adding `nosolve` and `nomap` in the ini file, one runs `ASOC.py my.ini`. SOC will make the radiation field simulation and save information about the absorbed energy into the file specified by the keyword `absorbed`.

2. **Solving of dust temperatures and emission** — One executes `A2E_MABU.py my.ini absorbed.data emitted.data` to compute the dust emission into (e.g.) the file `emitted.data`.

3. **Writing of emission maps** — One modifies the ini file by setting `iterations` 0 and removing `nomap`. The second SOC run `ASOC.py my.ini` will then produce the map files.

The script `ASOC_driver.py` automates the above. Thus, instead of the normal `ASOC.py my.ini` one can execute `ASOC_driver.py my.ini` and this should work irrespective of the number of dust components.

### Stochastically Heated Grains

Even if the calculations involved only a single dust component, if one solves the stochastic heating of the grains, the calculation of dust temperatures and emission cannot be done inside SOC. In fact, SOC only understands dust files of the simple type, listing absorption and scattering cross sections but no information about size distributions or enthalpies. Like in the case of multiple dust components above, the solution is to delegate the dust emission calculation to the `A2E_MABU.py` script.

In the case of stochastic heating, we need actually two descriptions of the dust model. The "simple" file is used by SOC. The temperature and emission calculations use the "gs" type, which includes those additional descriptions of size distributions and enthalpies.

The role of `A2E_MABU.py` is to loop over different dust components. Inside the script, the emission of each single dust component is computed with the script `A2E.py`. One could run `A2E.py` also directly, but that makes sense only if there is only one dust component, i.e. the absorptions are not divided between different dust components. The call to `A2E.py` takes the form:

```
A2E.py   some_dust.solver absorbed.data emitted.data
```

The first parameter `some_dust.solver` is a precomputed file (created automatically in the `ASOC_driver.py` script) that speeds up the actual calculations of the dust temperature distributions and dust emission (for this particular dust component). That file is prepared with yet another script `A2E_pre.py`:

```
A2E_pre.py some_dust.dust  freq.dat some_dust.solver
```

The full calculations thus involve quite a few steps:

1. Creation of the solver files for each dust component with stochastic heating
2. Creation of the "simple" dust files for use with SOC
3. Run of SOC to simulate the radiation field and save the result (e.g. the file `absorbed.data`)
4. Split the absorptions between the different dust components, run `A2E.py` to solve the emission for each of them, and sum the results to get the total emission
5. Run SOC another time to write the emission maps

Fortunately, these are already included in the script `ASOC_driver.py`. Thus, all one needs to do is to run `ASOC_driver.py my.ini`.

### Neural Networks for Stochastic Heating

Neural networks can be used to approximate dust emission based on information of absorptions at a small number of wavelengths. Especially in the case of stochastically heated grains, this provides a faster (but approximate) way to calculate the emission, after simulating the radiation field only at a few wavelengths. The use of neural networks with SOC requires the installation of the Pytorch machine learning library.

The calculations with neural networks include two steps. During the training run, one solves the emission from stochastically heated grains in the normal way. This provides the inputs for a neural network that then learns the mapping from the local radiation field intensity at the set of reference frequencies to the resulting dust emission at another set of output frequencies. During subsequent runs, the RT calculations can then be carried out using only the small set of reference frequencies, and the neural network then converts the information about the absorbed energy directly into estimates of the dust emission.

The neural network fit is mainly valid for the range of conditions found in the training set. Neural networks are able to give a good representation of the emission vs. absorption mapping already with a small number of connections, which suggests that also the training sets do not have to be very large. Training sets of tens of thousands of absorption-emission pairs might be enough, compared to the potentially millions of cells in the examined models.

When one is using neural networks, the ini file must specify both a set of the reference frequencies (probing the absorbed energy) and the frequencies for which the emission is to be predicted. The corresponding keywords are `nnabs` and `nnemit`. In the training run, the ini file must include the keyword `nnmake` to instruct SOC to train a neural network.

```
nnabs  0.3 0.6 1.0 2.0 5.0   # absorptions [um]
nnemit 3.4 4.6 25.0 100.0 250.0 500.0   # emission [um]
nnmake test nnthin 11
```

The radiation field is described through absorptions at five wavelengths between 0.3 µm and 5.0 µm. The wavelengths should be selected so that they give a good idea of the changes in the radiation field. Short wavelengths are sensitive to variations at the model surface and close to the radiation sources, longer wavelengths are better in describing fields at high optical depths. In this example, the trained network will predict emission at six wavelengths from 3.4 µm to 500 µm.

The argument of the keyword `nnthin` means that one will use only every 11th cell for the training. The default value is 1 (using every cell). If the value of `nnthin` is very large, the training set may fail to sample extreme conditions in the model, thus leading to less precise predictions for such regions.

When one is using an already trained network in further calculations, the keyword `nnmake` is replaced by `nnsolve`, followed by the string argument used with the keyword `nnmake` above:

```
nnabs   0.3  0.6  1.0  2.0  5.0
nnemit  3.4  4.6  25.0 100.0 250.0 500.0
nnsolve test
mapum   3.4  4.6  25.0 100.0 250.0 500.0
FITS    1
```

The keywords `nnabs` and `nnemit` must also be included, with the same list of wavelengths as when the library was created. During the run with `nnsolve`, only the wavelengths given by `nnabs` should be included in the radiative transfer calculations.

