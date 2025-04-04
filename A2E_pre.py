#!/usr/bin/env python

import os, sys

# We assume that the Python scripts and *.c kernel files are in this directory
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)
  
from   ASOC_aux import *
from   DustLib import *
import pyopencl as cl
import numpy as np
# import pickle
# from matplotlib.pylab import *

"""
Write solver.data using OpenCL.
"""

if (len(sys.argv)<4):
    print()
    print("Usage:   A2E_pre.py <gs-dustname>  <frequencyfile>  <solver-data-file> [NE]")
    print()
    print(" gset-dustname    =  name of the dust file gs_*.dust written by DustLib.py routine")
    print("                     write_A2E_dustfiles()")
    print(" frequencyfile    =  text file, one frequency per line")
    print(" solver-data-file =  output file <gs-dustname>.solver")
    print(" NE               =  optional parameter, number of enmthalpy bins (default is 256)")
    print()
    sys.exit()
    

DUST   =  GSETDust(sys.argv[1])
FREQ   =  asarray(loadtxt(sys.argv[2]), np.float32)
NFREQ  =  len(FREQ)
NSIZE  =  DUST.NSIZE
##TDOWN  =  np.zeros((NSIZE, NFREQ), np.float32)
## SCALE = 1.0e20  ---- replaced by FACTOR from ASOC_aux.py
Ef     =  PLANCK*FREQ
NE     =  256
if (len(sys.argv)>4):
    NE =  int(sys.argv[4])

NEPO   =  NE+1    # number of temperature bin boundaries
LOCAL  =  4
GLOBAL =  int((NE/64)+1)*64

# we do not want any np.zeros in CRT_SFRAC !!
DUST.CRT_SFRAC = clip(DUST.CRT_SFRAC, 1.0e-25, 1.0e30)

if (0):
    isize, freq = 3, 1.0e12
    print("SKabs(isize=3, f=1e12)     = %12.4e" %  DUST.SKabs(3, 1.0e12))
    print("SKabs_Int(isize=3, f=1e12) = %12.4e" %  DUST.SKabs_Int(3, 1.0e12))
    print("SKabs*SFRAC*GRAIN_DENSITY  = %12.4e" % (DUST.SKabs(3, 1.0e12)*DUST.GRAIN_DENSITY*DUST.CRT_SFRAC[isize]))
    sys.exit()


def PlanckIntensity(f, T):
    res = (2.0*PLANCK*(f/C_LIGHT)**2.0*f) / (exp(H_K*f/T) - 1.0)
    ## print(" B(%.1f) = %12.4e %12.4e %12.4e" % (T, res[1], res[20], res[39]))
    return res
    
    
if (1): # 2019-10-25
    TMIN = DUST.TMIN
    TMAX = DUST.TMAX

# ok, TMIN and TMAX vectors are the limits listed in <dust>.size
# print(TMIN, TMAX)
# sys.exit()

Ibeg  = np.zeros(NFREQ, np.int32)
L     = np.zeros(NE*NE, np.int32)

# start by calculating SKABS[NSIZE, NFREQ]
SKABS = np.zeros((NSIZE, NFREQ), float64)
for isize in range(NSIZE):
    SKABS[isize,:] =  DUST.SKabs_Int(isize, FREQ)  # ==  pi*a^2*Q*S_FRAC, including GRAIN_DENSITY


if (0):
    for isize in range(NSIZE):
        loglog(f2um(FREQ), SKABS[isize,:])        
    show(block=True)
    
        
# OpenCL initialisation --- kernel probably efficient only on CPU
context, queue = None, None
dev_found = False
sdevice = ''
try:
    sdevice = os.environ['OPENCL_SDEVICE']
except:
    sdevice = ''
    
for iplatform in range(5):
    try:  # get a GPU device
        platform  = cl.get_platforms()[iplatform]
        device    = platform.get_devices(cl.device_type.GPU)
        # print(' ...', device[0].name)
        if ('Oclgrind' in device[0].name): continue
        if (sdevice!=''):
            if (not(sdevice in device[0].name)): continue
        # device  = platform.get_devices(cl.device_type.GPU)
        context   = cl.Context(device)
        queue     = cl.CommandQueue(context)
        dev_found = True
        # print(' ... %s ok' % (device[0].name))
        break
    except:
        pass
if (dev_found==False): # just as backup, pick a CPU device
    print("GPU not found, trying to find CPU...")
    for iplatform in range(5):
        try:
            platform  = cl.get_platforms()[iplatform]
            device    = platform.get_devices(cl.device_type.CPU)
            print(' ...', device[0].name)
            if ('Oclgrind' in device[0].name): continue
            if (sdevice!=''):
                if (not(sdevice in device[0].name)): continue
            # device  = platform.get_devices(cl.device_type.GPU)
            context   = cl.Context(device)
            queue     = cl.CommandQueue(context)
            dev_found = True
            # print(' ... %s ok ' % (device[0].name))            
            break
        except:
            pass
print("A2E_pre.py using device", device[0].name)

src       =  open(INSTALL_DIR+"/kernel_A2E_pre.c").read()
program   =  cl.Program(context, src).build(" -D FACTOR=%.4ef " % FACTOR)
mf        =  cl.mem_flags 
FREQ_buf  =  cl.Buffer(context, mf.READ_ONLY,  4*NFREQ)
Ef_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*NFREQ)
SKABS_buf =  cl.Buffer(context, mf.READ_ONLY,  4*NFREQ)  # one size!
E_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*NEPO)
T_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*NEPO)
Tdown_buf =  cl.Buffer(context, mf.READ_WRITE, 4*NE)
L1_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NE*NE)
L2_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NE*NE)
Iw_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NE*NE*NFREQ)      # may be ~100 MB !
wrk_buf   =  cl.Buffer(context, mf.READ_WRITE, 4*(NE*NFREQ+NE*(NFREQ+4)))
noIw_buf  =  cl.Buffer(context, mf.WRITE_ONLY, 4*(NE-1))     # for each l = lower

Tdown     =  np.zeros(NE, np.float32)
wrk       =  np.zeros(NE*NFREQ+NE*(NFREQ+4), np.float32)
Iw        =  np.zeros(NE*NE*NFREQ, np.float32)
noIw      =  np.zeros(NE-1, np.int32)
EA        =  np.zeros((NFREQ, NE), np.float32)

cl.enqueue_copy(queue, FREQ_buf, FREQ)                    # NFREQ
cl.enqueue_copy(queue, Ef_buf,   asarray(Ef, np.float32))    # just for convenience

PrepareTdown =  program.PrepareTdown
PrepareTdown.set_scalar_arg_dtypes([np.int32, None, None, None, np.int32, None, None, None])

PrepareTdown2 =  program.PrepareTdown2
PrepareTdown2.set_scalar_arg_dtypes([np.int32, None,       None, np.int32, None, None, None])

# if (0):
#     #    first method -- not for large grains (some problem...)
#     PrepareIw    =  program.PrepareIntegrationWeights
# if (0):
#     #    almost identical to the above (not for large grains)
#     PrepareIw    =  program.PrepareIntegrationWeightsGD

if (1):
    #  2024-12-22 this is the only version one that works for large grains
    #  and arbitrary (f, E) grids. The above two are ok but only if large grains
    #  are handled via equilibrium temperature calculations (there
    #  amin_equ=0.02 in write_A2E_dustfiles() worked ok with normal ISRF)
    PrepareIw    =  program.PrepareIntegrationWeightsTrapezoid
    
PrepareIw.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None, None, None, None, None])

# Start writing the solver.data file
fp = open(sys.argv[3], "wb")
asarray( [NFREQ,    ], np.int32   ).tofile(fp)
asarray(  FREQ,        np.float32 ).tofile(fp)
asarray( [DUST.GRAIN_DENSITY,] , np.float32).tofile(fp)
asarray( [NSIZE,],     np.int32   ).tofile(fp)
# 2021-04-26 -- add DUST.SIZE_A to the solver file @@
asarray( DUST.SIZE_A, np.float32  ).tofile(fp)
# Dustlib CRT_SFRAC is the fraction multiplied with GRAIN_DENSITY
asarray(  DUST.CRT_SFRAC/DUST.GRAIN_DENSITY, np.float32 ).tofile(fp)  # this SFRAC is one with sum(SFRAC)==1, not GRAIN_DENSITY
asarray( [NE,       ], np.int32   ).tofile(fp)
asarray(  SKABS,       np.float32 ).tofile(fp)   # SKAbs_Int()    ~     pi*a^2*Qabs * SFRAC, including GRAIN_DENSITY

# Below PrepareTdown uses SKabs() instead of SKAbs_Int() => divide by SFRAC  (including GRAIN_DENSITY)

# in the following Tdown needs SKABS ... but per grain => need to divide CRT_SFRAC
# print("NSIZE %d, NFREQ %d, NE %d" % (NSIZE, NFREQ, NE))

if (1): # save also temperature grids for each size... auxiliary file
    fpX = open('%s.tgrid' % (sys.argv[3].replace('.solver','')), 'wb')
    asarray([NSIZE, NEPO], np.int32).tofile(fpX)
else:
    fpX = None
    
for isize in range(NSIZE):    
    
    # A2ELIB used logarithmically spaced energies
    # print("--------------------------------------------------------------------------------")
    emin  =  DUST.T2E(isize, TMIN[isize])
    emax  =  DUST.T2E(isize, TMAX[isize])
    # print("nsize [%3d] %9.4f um  E %.3e - %.3e  T = %5.2f - %7.2f K" % (isize, DUST.SIZE_A[isize]*1.0e4, emin, emax, TMIN[isize], TMAX[isize]))


    if (0): #
        E     =  exp(log(emin)+(arange(NEPO)/float(NE))*(log(emax)-log(emin))) # NEPO elements
        T     =  DUST.E2T(isize, E)       # NEPO !
    elif(1):
        # GSETDustO.cpp has --  e = TemperatureToEnergy_Int(s, TMIN[s]+(TMAX[s]-TMIN[s])* pow(ie/(NEPO-1.0), 2.0)) ;
        T   =  TMIN[isize]+(TMAX[isize]-TMIN[isize])* (arange(NEPO)/(NEPO-1.0))**2.0
        E   =  DUST.T2E(isize, T)
    elif(0):
        T   =  np.logspace(log10(TMIN[isize]), log10(TMAX[isize]), NEPO)
        E   =  DUST.T2E(isize, T)
    else:
        T   =  TMIN[isize]+(TMAX[isize]-TMIN[isize])* (arange(NEPO)/(NEPO-1.0))**3.5
        E   =  DUST.T2E(isize, T)



        
    cl.enqueue_copy(queue, SKABS_buf, asarray((SKABS[isize,:]/DUST.CRT_SFRAC[isize]),np.float32))
    cl.enqueue_copy(queue, E_buf, asarray(E, np.float32))  # E[NEPO]
    cl.enqueue_copy(queue, T_buf, asarray(T, np.float32))  # T[NEPO]  --- NEPO elements for interpolation in PrepareTdown

    
    if (fpX):
        asarray(T, np.float32).tofile(fpX)   #   T[NEPO]
        asarray(E, np.float32).tofile(fpX)   #   E[NEPO]
    
    # PrepareIntegrationWeights() kernel
    PrepareIw(queue, [GLOBAL,], [LOCAL,], NFREQ, NE, Ef_buf, E_buf, L1_buf, L2_buf, Iw_buf, wrk_buf, noIw_buf)
    cl.enqueue_copy(queue, Iw,   Iw_buf)
    cl.enqueue_copy(queue, noIw, noIw_buf)      # one worker = one l=lower bin ~ at most NE*NFREQ Iw weights
    sum_noIw = np.sum(noIw)                     # number of actual integration weights, for each l = lower bin
    asarray([sum_noIw,], np.int32).tofile(fp)   # --> noIw
    for l in range(0, NE-1):                    # loop over lower bins = results of each kernel worker
        ind = l*NE*NFREQ                        # start of the array reserved for each l = each worker
        asarray(Iw[ind:(ind+noIw[l])], np.float32).tofile(fp)     # --> Iw, for l
    cl.enqueue_copy(queue, L,  L1_buf)
    L[0] = -2
    asarray(L, np.int32).tofile(fp)              # --> L1
    cl.enqueue_copy(queue, L,  L2_buf)
    L[0] = -2
    asarray(L, np.int32).tofile(fp)              # --> L2

    if (1):
        # PrepareTdown() kernel
        PrepareTdown(queue,  [GLOBAL,], [LOCAL,], NFREQ, FREQ_buf, Ef_buf, SKABS_buf, NE, E_buf, T_buf, Tdown_buf)
    else:  #  again confirmed 2025-04-04 -- PrepareTdown() and PrepareTdown2() give identical emission.
        # the same results ...
        PrepareTdown2(queue, [GLOBAL,], [LOCAL,], NFREQ, FREQ_buf,         SKABS_buf, NE, E_buf, T_buf, Tdown_buf)

        
    cl.enqueue_copy(queue, Tdown, Tdown_buf)    # --> Tdown
    ### Tdown /= DUST.CRT_SFRAC[isize]   # GRAIN_DENSITY already in CRT_SFRAC !! --- division already in SKABS
    Tdown.tofile(fp)
    # print('T %.3e - %.3f  E %.3e - %.3e   Ef %.3e - %.3e' % (T[0], T[-1], E[0], E[-1], Ef[0], Ef[-1]))
    # print('SKABS ', SKABS[isize, 0:5], SKABS[isize, -5:])
    # print(Tdown[0:4])

    #if (isize==5):
    #    OPT = { 'E': E,  'T': T,  'FREQ': FREQ, 'NE': NE,
    #            'SKABS' : asarray((SKABS[isize,:]/DUST.CRT_SFRAC[isize]),np.float32),
    #            'Tdown' : Tdown }
    #    fpdump = open('SHG_TST.dump', 'wb')
    #    pickle.dump(OPT, fpdump)
    #    fpdump.close()

    
    # Prepare EA[ifreq, iE]  <---- storage order!
    TC  =  DUST.E2T(isize, 0.5*(E[0:NE]+E[1:])) # NE temperatures for the *centre* of each energy bin
    for iE in range(NE):                        # EA[ifreq, iE], here SKABS is still pi*a^2*Qabs*GD*SFRAC
        EA[:, iE] =  SKABS[isize,:] * (PlanckIntensity(asarray(FREQ, float64), TC[iE]) / (PLANCK*FREQ))
    EA *= FACTOR*4.0*np.pi
    asarray(EA, np.float32).tofile(fp)             # --> EA

    # Prepare Ibeg array
    for ifreq in range(NFREQ):
        startind = 1 
        while((0.5*(E[startind-1]+E[startind])<Ef[ifreq]) & (startind<NEPO-1)): startind += 1
        Ibeg[ifreq] = startind
    Ibeg.tofile(fp)                                # --> Ibeg
                                                       
fp.close()

if (fpX): fpX.close()


