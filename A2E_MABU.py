#!/usr/bin/env python

import os, sys, time
# from MJ.mjDefs import *

# we assume that the Python scripts and *.c kernel files are in this directory
# HOMEDIR = os.path.expanduser('~/')
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)

from ASOC_aux    import *
try:
    from ASOC_aux_NN import *
except:
    print("pytorch probably not installed => NN (neural net) absorption -> emission mapping not possible")
    
# temporary directories for absorption and emission files
# emission files may be smaller (if output frequencies are limited)
# and easier to put to ram disk
ASHAREDIR = '/dev/shm/'     # temporary directory for absorptions
ESHAREDIR = '/dev/shm/'     # temporary directory for emissions

KEEP_COMPONENT_EMITTED = False  # leave emitted files on the disk for each dust component spearately
USE_MMAP = False                # False, unless CELLS*NOFREQ does not fit to main memory !!

if (len(sys.argv)<2):
    print("Usage:")
    print("      A2E_MABU.py  ini absorbed.data emitted.data [ofreq.dat] ")
    print("Input:")
    print("  soc.ini        =  SOC ini file (listing the dusts and abundances etc.)")
    print("  absorbed.data  =  absorptions  [CELLS, nfreq] or [CELLS, nlfreq]")
    print("  emitted.data   =  solved emissions [CELLS, nfreq]")
    print("Note on the dust names:")
    print("   If this is called by A2E_driver.py, the ini file may contain dust names")
    print("   like gs_aSilx.dust. However, the solver file will not have the gs_ prefix")
    print("   => any gs_ prefixes of dust names will be dropped when reading the ini.\n")
    print("Note on mapum:")
    print("   mapum keyword used to be taken into account, limiting the frequencies in the")
    print("   emitted file -- however, mapum is now ignored, for compatibility with ASOC_driver.py")
    print("")
    print("2023-12-30 -- include NN fit and NN solution of emission")
    print("   ini file should contain keywords nnabs for absorbed and nnemit for the ")
    print("       emitted wavelengths in the NN solution")
    print("   nnmake prefix   =>  make NN mappings <prefix>_<dust>.nn")
    print("   nnthin #        =>  in the above, use only thin:th cell")
    print("   nnsolve prefix  =>  calculate emission using NN mapping only,")
    print("                       absorbed.data must contain nnabs only, ")
    print("                       emitted.data will contain nnemit only")
    print("                       == A2E_driver.py must take these into account")
    print("                          ini RT run (nnabs only) and map writing (nnemit only)")
    print("                       ... as well as calling A2E_MABU with absorbed.data")
    print("                           truncated to nnabs only")
    print("2024-04-02 -- added CR_HEATING flag to ini file")
    print("                    0 = no extra heating,     1 = default CR heating, ")
    print("                    2 = 2x default CR heaing, 3 = ad hoc gas-dust heating for dust")
    sys.exit()

    
F_OFREQ =  ""
for arg in sys.argv[4:]:
    F_OFREQ = arg

fplog = open('A2E_MABU.log', 'w')
fplog.write("[%s]\n\n" % time.asctime())
fplog.write("A2E_MABU.py")
for i in range(len(sys.argv)): fplog.write(" %s" % sys.argv[i])
fplog.write("\n")


# find dusts and potential abundance file names from the ini
# DUST[] will contain only names of simple dusts !!
DUST, AFILE, EQDUST = [], [], []
UM_MIN, UM_MAX      = 0.00001, 999999.0
MAP_UM    = []     # list of wavelengths for which emission calculated, MAP_UM==[] => all frequencies
GPU       = 0
platforms = []
AALG      = None   # will have dust and a_alg file name

CR_HEATING  =  0
CLOUD       =  None
KDENSITY    =  1.0

nnabs, nnemit, nnmake, nnsolve, nnthin, nngpu = [], [], '', '', 1, 0
# nnthin  = 1    # not used here, only in A2E_driver


fp  = open(sys.argv[1], 'r')
for line in fp.readlines():
    s = line.split()
    if (len(s)<2): continue
    if (s[0]=='CR_HEATING'):
        # only for values 0, 1, and 2 A2E_MABU uses the parameter the same as ASOC == as CR scaling factor
        CR_HEATING = int(s[1])
        print("CR_HEATING %d !!!" % CR_HEATING)
        #   CR_HEATING=1  =>  single dust, full CR heating to dust heating
        #   CR_HEATING=2  =>  multiple dusts, full CR heating divided between dust populations
        #   CR_HEATING=3  =>  multiple dusts, heating from dust-gas coupling with ad hoc gas temperature + temp. difference
        time.sleep(2)
    if (s[0][0:1]=='#'): continue
    if (s[0]=='cloud'):
        CLOUD = s[1]
    if (s[0]=='density'):
        KDENSITY = float(s[1])
    if (s[0].find('remit')==0):
        UM_MIN = float(s[1])
        UM_MAX = float(s[2])
    if (s[0].find('device')>=0):
        if (s[1].find('g')>=0): GPU = 1
        else:                   GPU = 0
    if (s[0].find('platform')>=0):
        platforms = [ int(s[1]), ]    # user requested a specific OpenCL platform
    if (s[0][0:6]=='optica'):
        dustname = s[1]
        tag      = open(dustname).readline().split()[0]
        if (tag=='eqdust'):
            DUST.append(dustname)  # full filename for equilibrium dusts
            EQDUST.append(1)
        else:                      # else SHG = gset dust
            dustname = s[1].replace('_simple.dust','')
            # drop "gs_" from the dust name -- the solver file will be named without "gs_"
            if (dustname[0:3]=='gs_'):                  
                dustname = dustname[3:]
            DUST.append(dustname.replace('.dust', ''))  # dust basename
            EQDUST.append(0)
        if (len(s)>2):      # we have abundance file
            if (s[2]=="#"):
                AFILE.append("")
            else:
                AFILE.append(s[2])
        else:
            AFILE.append("")
    if (s[0][0:5]=='XXXXXXmapum'):  # list of output frequencies --- NO ==> use MAP_UM only for map writing ??
        # this is used in the SolveEquilibriumDust to select a subset of frequencies to output emission file
        # with stochastic heating -- handled by a call to A2E.py --- one can currently select just a single freq.
        if (0):  # no --- use MAP_UM ***only*** when writing maps
            for tmp in s[1:]:
                MAP_UM.append(float(tmp))
            MAP_UM = asarray(MAP_UM, float32)
    # keyword polarisation to indicate the aalg file for a dust component
    # => one will save the emission in the normal way but one also saves
    # the polarised emission to a separate file, this dust component, a>aalg
    if (s[0][0:6]=='polari'): # arguments = dust name, aalg file name
        if (len(s)<3):
            print("Error in ini: must have polarisation dust_name aalg_file_name"), sys.exit(0)
        if (AALG):
            AALG.update({ s[1].replace('.dust', ''): s[2] })
        else:
            AALG  = { s[1].replace('.dust', ''): s[2] }
            
    if (s[0].find('nnabs')>=0):       
        nnabs  =  get_floats(s[1:])     # wavelengths included in absorptions (NN mapping)
        nnabs  =  np.sort(um2f(nnabs))  # frequencies in increasing order
    if (s[0].find('nnemit')>=0):
        nnemit =  get_floats(s[1:])     # wavelengths included in emission (NN mapping)
        nnemit =  np.sort(um2f(nnemit)) # frequencies in increasing order
    if (s[0].find('nnmake')>=0):        # make => using full sets of input and output frequencies
        nnmake = s[1]                   #  prefix for  <prefix>_<dust>.nn 
    if (s[0].find('nnsolve')>=0):       # solve  =   nnabs -> nnemit wavelengths only
        nnsolve = s[1]
    if (s[0].find('nnthin')>=0):        # A2E_driver has taken care of nnthin already
        nnthin = 1                      # in absorbed.data
    if (s[0].find('nngpu')>=0):         # use GPU for NN calculations
        nngpu  = int(s[1])
    # nnmake  =>  full sets of frequencies on absorbed, in dusts
    # nnsolve =>  absorbed.data only for nnabs, emitted.data only for nnemit wavelengths
    


    
fp.close()
fplog.write("%s ->  GPU=%d\n" % (sys.argv[1], GPU))

print("================================================================================")
print("A2E_MABU dusts: ", DUST)
print("nnmake  ", nnmake)
print("nnsolve ", nnsolve)
print("nnthin  ", nnthin)
if (len(nnabs)>0):  print("nnabs   ", f2um(nnabs))
if (len(nnemit)>0): print("nnemit  ", f2um(nnemit))
print("================================================================================")


# GPU is passed onto A2E.py on command line .... 1.3 would mean GPU and platform 3 !!!
GPU_TAG = 0
if (GPU>0):                 # GPU was specified on command line or in the ini file
    if (len(platforms)>0):  #  [] or one value read from the ini file
        GPU_TAG = 1.0 + 0.1*platforms[0]  # encoded into float for the A2E.py command line argument
    else:
        GPU_TAG = 1         # platform not specified
# we need platforms also in this script (solving of equilibrium temperature dust)
if (len(platforms)<1):
    platforms = arange(5)
    
    

fplog.write("DUST\n")
for x in DUST:
    fplog.write("    %s\n" % x)   # this will be aSilx ... not gs_aSilx or aSilx_simple
fplog.write("EQDUST\n")
for x in EQDUST:
    fplog.write("    %s\n" % x)


            
# read KABS for each dust from the solver files
# in case of nnsolve, KABS <-  KABS[IND_nnabs] !!
NDUST = len(DUST)
GD    = np.zeros(NDUST, np.float32)  # grain density
RABS  = []
FREQ  = []
NFREQ = 0
for idust in range(NDUST):
    print("=== dust %d: %s" % (idust, DUST[idust]))
    if (EQDUST[idust]):
        print("=== EQDUST")
        lines      =  open(DUST[idust]).readlines()
        GD[idust]  =  float(lines[1].split()[0])
        radius     =  float(lines[2].split()[0])
        d          =  np.loadtxt(DUST[idust], skiprows=4)
        NFREQ      =  d.shape[0]
        FREQ       =  d[:,0]
        kabs       =  np.pi*radius**2.0*GD[idust] * d[:,2]
        if (len(RABS)<2):
            RABS = np.zeros((NFREQ, NDUST), np.float64)  # total cross section per unit density
        RABS[:,idust] = kabs
    else:
        # Stochastically heated grains
        print("=== SHG")
        fp     = open('%s.solver' % DUST[idust], 'rb') # solver will be called aSilx.solver, not gs_aSilx.solver...
        NFREQ  = np.fromfile(fp, np.int32, 1)[0]
        print("A2E_MABU ==>  %30s  HAS %3d FREQUENCIES" % (DUST[idust], NFREQ))
        ## sys.exit()
        FREQ   = np.fromfile(fp, np.float32, NFREQ)
        GD     = np.fromfile(fp, np.float32, 1)[0]
        NSIZE  = np.fromfile(fp, np.int32, 1)[0]
        SIZE_A = np.fromfile(fp, np.float32, NSIZE)    # 2021-05-06 added !!!!!!!!!!!!
        S_FRAC = np.fromfile(fp, np.float32, NSIZE)
        NE     = np.fromfile(fp, np.int32, 1)[0]
        SK_ABS = np.fromfile(fp, np.float32, NSIZE*NFREQ).reshape(NSIZE,NFREQ)  # Q*pi*a^2 * GD*S_FRAC
        SK_ABS = np.asarray(SK_ABS, np.float64)
        fp.close()
        if (len(RABS)<2):
            RABS = np.zeros((NFREQ, NDUST), np.float64)    # relative absorption per grain population
        RABS[:,idust] = sum(SK_ABS, axis=0)                # total cross section as sum over sizes
        fp.close()

        
# for NN runs, get indices into FREQ for nnabs and for nnemit
IND_nnabs, IND_nnemit = [], []
if (len(nnabs)>0):
    IND_nnabs = zeros(len(nnabs), int32)   # indices of absorbed.dat frequencies in NN runs
    for i in range(len(nnabs)):
        f = nnabs[i]
        k = argmin(abs(f-FREQ))
        if ((abs(f-FREQ[k])/f)>0.02):
            print("*** Error in A2E_MABU: nnabs %.3f does not correspond to any frequency" % nnabs[i])
            sys.exit()
        IND_nnabs[i] = k  #    FREQ[IND_nnabs[i]] ~ nnabs[i]
if (len(nnemit)>0):
    IND_nnemit = zeros(len(nnemit), int32)
    for i in range(len(nnemit)):
        f = nnemit[i]
        k = argmin(abs(f-FREQ))
        if ((abs(f-FREQ[k])/f)>0.02):
            print("*** Error in A2E_MABU: nnemit %.3f does not correspond to any frequency" % nnemit[i])
            sys.exit()
        IND_nnemit[i] = k  #    FREQ[IND_nnemit[i]] ~ nnemit[i]
        
if (0):
    clf()
    imshow(log10(RABS), aspect='auto')
    colorbar()
    savefig('RABS.png')
    show()
    sys.exit()

    
    

NOFREQ = NFREQ   # NOFREQ = number of output frequencies
# ini may have limited output frequencies using the keyword remit --- eqdusts only??
if (len(MAP_UM)>0): # this overrides [UM_MIN, UM_MAX]
    NOFREQ =  len(MAP_UM)
else:
    m      =  np.nonzero((FREQ<=um2f(UM_MIN))&(FREQ>=um2f(UM_MAX)))
    NOFREQ =  len(FREQ[m])
print('NOFREQ LIMITED TO %d --- currently only for eqdust solved inside A2E_MABU.py itself !!!' % NOFREQ)

    



# file of absorptions should start with three integers
H             =  np.fromfile(sys.argv[2], np.int32, 2)
CELLS, NFFREQ = H[0], H[1]
print("=== Absorption file:  CELLS %d, NFFREQ %d, NFREQ %d, NDUST %d" % (CELLS, NFFREQ, NFREQ, NDUST))

# Convert RABS[NFREQ, NDUST] to better normalised relative cross section
RABS    = clip(RABS, 1.0e-40, 1.0e30)   # must not be zero 
for ifreq in range(RABS.shape[0]):
    RABS[ifreq,:]  /= (1.0e-40+sum(RABS[ifreq,:]))
RABS    =  np.clip(RABS, 1.0e-30, 1.0)                       # RABS[freq, dust]


if (len(nnsolve)>0): 
    # if one is solving emission with NN, the ini file is still for the full set of frequencies
    # => extract to RABS only the absorption coeffcients elements that correspond to nnabs frequencies
    #    RABS[freq_absorbed, dust]
    #print('\n\n\n\n\n')
    #clf()
    #plt.loglog(f2um(FREQ), RABS[:,0], 'b-')
    #plt.loglog(f2um(FREQ), RABS[:,1], 'r-')
    RABS = RABS[IND_nnabs,:].copy()   # original RABS = RABS[NFREQ, NDUST]
    print("RABS = ABSORPTIONS FOR NN CALCULATION, WAVELENGTH")
    print(f2um(FREQ)[IND_nnabs])
    #plt.loglog(f2um(FREQ[IND_nnabs]), RABS[:,0], 'bx')
    #plt.loglog(f2um(FREQ[IND_nnabs]), RABS[:,1], 'rx')
    #plt.show(block=True)
    #sys.exit()

    
# print("=== A2E_MABU.py .... NFREQ %d" % NFREQ)
fplog.write("NFREQ = %d\n" % NFREQ)


C_LIGHT =  2.99792458e10  
PLANCK  =  6.62606957e-27 
H_K     =  4.79924335e-11 
D2R     =  0.0174532925       # degree to radian
PARSEC  =  3.08567758e+18 
H_CC    =  7.372496678e-48 


def PlanckSafe(f, T):  # Planck function
    # Add clip to get rid of warnings
    return 2.0*H_CC*f*f*f / (np.exp(np.clip(H_K*f/T,-100,+100))-1.0)


def opencl_init(GPU, platforms, verbose=False):
    """
    Initialise OpenCL environment.
    """
    print("=== opencl_init === GPU, platforms ", GPU, platforms)
    platform, device, context, queue = None, None, None, None
    ok = False
    # print("........... platforms ======", platforms)
    sdevice = ''
    try:
        sdevice = os.environ['OPENCL_SDEVICE']
    except:
        sdevice = ''
    ####
    for iii in range(2):
        for iplatform in platforms:
            tmp = cl.get_platforms()
            if (verbose):
                print("--------------------------------------------------------------------------------")
                print("GPU=%d,  TRY PLATFORM %d" % (GPU, iplatform))
                print("NUMBER OF PLATFORMS: %d" % len(tmp))
                print("PLATFORM %d = " % iplatform, tmp[iplatform])
                print("DEVICE ",         tmp[iplatform].get_devices())
                print("--------------------------------------------------------------------------------")
            try:
                platform  = cl.get_platforms()[iplatform]
                if (GPU):
                    device  = platform.get_devices(cl.device_type.GPU)
                else:
                    device  = platform.get_devices(cl.device_type.CPU)
                if ('Oclgrind' in device[0].name):
                    device = []
                elif (len(sdevice)>0):
                    if (not(sdevice in device[0].name)):
                        device = []
                context  = cl.Context(device)
                queue    = cl.CommandQueue(context)
                ok       = True
                if (verbose): print("    ===>     DEVICE ", device, " ACCEPTED !!!")
                break
            except:                
                if (verbose): print("    ===>     DEVICE ", device, " REJECTED !!!")                
                pass
        if (ok):
            return context, queue, cl.mem_flags
        else:
            if (iii==0):
                platforms = arange(4)  # try without specific choise of platform
            else:
                print("*** A2E_MABU => opencl_ini could not find valid OpenCL device *** ABORT ***")
                time.sleep(10)
                sys.exit()
            



def SolveEquilibriumDust(dust, f_absorbed, f_emitted, UM_MIN=0.0001, UM_MAX=99999.0, MAP_UM=[], \
    GPU=False, platforms=[0,1,2,3,4], AALG=None):
    """
    Calculate equilibrium temperature dust emission based on absorptions.
    Input:
        dust            =   name of the dust file (type eqdust)
        f_absorbed      =   file name for absorptions = CELLS, NFREQ, floats[CELLS*NFREQ]
        f_emitted       =   file name for emissions   = CELLS, NFREQ, floats[CELLS*NFREQ]
        UM_MIN, UM_MAX  =   limits output frequencies
        MAPUM           =   optional list of output frequencies
        GPU             =   if True, use GPU instead of CPU
        platforms       =   OpenCL platforms, default [0, 1, 2, 3, 4]
        AALG            =   optional, a_alg files for individual dust components
    Note:
        2021-04-26, added AALG
        One does not have emission separately for grains of different size. However,
        even if we assume that all grains are at the same temperature irrespective of their
        size, we will at least have temperature separately for each dust component.
        That is, R is not based only on KABS but based on KABS*B(T), taking into account the
        different T of Si and Gr. 
        If AALG is given:
            AALG[dust] is file containing a_alg for each cell
            in addition to <f_emitted>, we also write <f_emitted>.P that is emission times R,
            R is obtained from <dust>.rpol_single, scaling each emission with R(a_alg).
            <dust>.rpol is  R[a, freq], first row has frequencies, first column a
            <dust>.rpol_single is KABS(aligned)/KABS where KABS is only for this dust!!
            Note that write_simple_dust_pol() can be used with a set of dust species - when
            its input parameter tmp.dust is the sum of opacities over all dust species =>
            <dust>.rpol is usually KABS(aligned)/KABS_TOT with KABS_TOT sum over species. 
            ***BUT* --- Here write_simple_dust_pol() must have been run with the
            current dust species (Si) only, using filename=Si.dust, i.e. a simple dust file
            for the Si dust species only => R should have first rows = smallest a_alg  values
            equal to one !!! Therefore, we use file names <dust>.rpol_single
            instead of the normal <dust>.rpol.
    """
    # Read dust data
    print("============================================================")
    print("      SolveEquilibriumDust(%s), GPU =" % dust, GPU)
    print("============================================================")
    # fplog.write("      SolveEquilibriumDust(%s)\n" % dust)
    lines  =  open(dust).readlines()
    gd     =  float(lines[1].split()[0])
    gr     =  float(lines[2].split()[0])
    d      =  np.loadtxt(dust, skiprows=4)
    FREQ   =  np.asarray(d[:,0].copy(), np.float32)
    KABS   =  np.asarray(d[:,2] * gd * np.pi*gr**2.0, np.float32)   # cross section PER UNIT DENSITY
    # Start by making a mapping between temperature and energy
    NE     =  30000
    TSTEP  =  1600.0/NE    # hardcoded upper limit 1600K for the maximum dust temperatures
    TT     =  np.zeros(NE, np.float64)
    Eout   =  np.zeros(NE, np.float64)
    DF     =  FREQ[2:] - FREQ[:(-2)]  #  x[i+1] - x[i-1], lengths of intervals for Trapezoid rule
    # Calculate EMITTED ENERGY per UNIT DENSITY, scaled by 1e20 -> FACTOR
    for i in range(NE):
        TT[i]   =  1.0+TSTEP*i
        TMP     =  KABS * PlanckSafe(np.asarray(FREQ, np.float64), TT[i])
        # Trapezoid integration TMP over freq frequencies
        res     =  TMP[0]*(FREQ[1]-FREQ[0]) + TMP[-1]*(FREQ[-1]-FREQ[-2]) # first and last step
        res    +=  sum(TMP[1:(-1)]*DF)          # the sum over the rest of TMP*DF
        Eout[i] =  (4.0*np.pi*FACTOR) * 0.5 * res  # energy corresponding to TT[i] * 1e20, per unit density
    # Calculate the inverse mapping    Eout -> TTT
    Emin, Emax  =  Eout[0], Eout[NE-1]*0.9999
    print("      Mapping EOUT ==>   Emin %12.4e, Emax %12.4e\n" % (Emin, Emax))
    # E ~ T^4  => use logarithmic sampling
    kE          =  (Emax/Emin)**(1.0/(NE-1.0))  # E[i] = Emin*pow(kE, i)
    oplgkE      =  1.0/np.log10(kE)
    # oplgkE = 1.0/log10(kE)
    ip          =  interp1d(Eout, TT)           # (linear) interpolation from energy to temperature
    TTT         =  np.asarray(ip(Emin * kE**np.arange(NE)), np.float32)
    # Set up kernels
    CELLS, NFREQ=  np.fromfile(f_absorbed, np.int32, 2)
    context, commands, mf = opencl_init(GPU, platforms)
    source      =  open(INSTALL_DIR+"/kernel_eqsolver.c").read()
    ARGS        =  "-D CELLS=%d -D NFREQ=%d -D FACTOR=%.4ef -D CR_HEATING=%d" % (CELLS, NFREQ, FACTOR, CR_HEATING)
    program     =  cl.Program(context, source).build(ARGS)
        
    # Use the E<->T  mapping to calculate ***TEMPERATURES** on the device
    GLOBAL      =  32768
    LOCAL       =  [8, 32][GPU]
    kernel_T    =  program.EqTemperature
    #                               icell     kE          oplgE       Emin        NE         FREQ   TTT   ABS   T
    kernel_T.set_scalar_arg_dtypes([np.int32, np.float32, np.float32, np.float32, np.int32 , None,  None, None, None])
    FREQ_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=FREQ)
    TTT_buf     =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=TTT)
    KABS_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=KABS)
    ABS_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*GLOBAL*NFREQ)  # batch of GLOBAL cells
    T_buf       =  cl.Buffer(context, mf.READ_WRITE, 4*CELLS)
    EMIT_buf    =  cl.Buffer(context, mf.WRITE_ONLY, 4*CELLS)
    # Solve temperature GLOBAL cells at a time
    TNEW        =  np.zeros(CELLS, np.float32)
    tmp         =  np.zeros(GLOBAL*NFREQ, np.float32)
    # Open file containing absorptions
    FP_ABSORBED =  open(f_absorbed, 'rb')
    CELLS, NFREQ=  np.fromfile(FP_ABSORBED, np.int32, 2) # get rid of the header
    print("      SolveEquilibriumDust(%s): CELLS %d, NFREQ %d" % (f_absorbed, CELLS, NFREQ))
    t0          =  time.time()
    for ibatch in range(int(CELLS/GLOBAL+1)):
        a   =  ibatch*GLOBAL
        b   =  min([a+GLOBAL, CELLS])  # interval is [a,b[
        # no need to use mmap file for absorptions - values are read in order
        # print("      Solving  eqdust [%6d, %6d[ OUT OF %6d" % (a, b, CELLS))
        tmp[0:((b-a)*NFREQ)] =  np.fromfile(FP_ABSORBED, np.float32, (b-a)*NFREQ)
        cl.enqueue_copy(commands, ABS_buf, tmp)
        kernel_T(commands, [GLOBAL,], [LOCAL,], a, kE, oplgkE, Emin, NE, FREQ_buf, TTT_buf, ABS_buf, T_buf)
    cl.enqueue_copy(commands, TNEW, T_buf)    
    a, b, c  = np.percentile(TNEW, (10.0, 50.0, 90.0))
    print("      Solve temperatures: %.2f seconds =>  %10.3e %10.3e %10.3e" % (time.time()-t0, a,b,c))
    FP_ABSORBED.close()
    
    if (0):
        np.asarray(TNEW, np.float32).tofile(ESHAREDIR+'/TNEW.bin')
    else:
        np.asarray(TNEW, np.float32).tofile('%s.T' % dust)

        
    # Use another kernel to calculate ***EMISSIONS*** -- per unit density (and abundance)
    # 2019-02-24 --- this can be restricted to output frequencies [UM_MIN, UM_MAX]
    # 2021-05-01 --- MAP_UM list overrides this, if given
    kernel_emission = program.Emission
    #                                      FREQ        KABS         T     EMIT
    kernel_emission.set_scalar_arg_dtypes([np.float32, np.float32,  None, None ])
    GLOBAL   =  int((CELLS/LOCAL+1))*LOCAL
    if ((GLOBAL%64)!=0): GLOBAL = int((GLOBAL/64+1))*64
    # figure out the actual output frequencies
    MOUT   = np.nonzero(FREQ>0.0)   # MOUNT = by default all the frequencies
    if (len(MAP_UM)>0):             # MAP_UM overrides [UM_MIN, UM_MAX] interval
        df       =  (FREQ-um2f(MAP_UM))/FREQ
        MOUT     =  nonzero(abs(df)<0.001)
        # print("MAP_UM given => MOUT", MOUT)
    else:
        MOUT     =  np.nonzero((FREQ<=um2f(UM_MIN))&(FREQ>=um2f(UM_MAX)))
        # print("UM_MIN %.2f, UM_MAX %.2f => MOUT " % (UM_MIN, UM_MAX), MOUT)
    ofreq  = FREQ[MOUT]
    nofreq = len(ofreq)
    # Solve emission one frequency at a time, all cells on a single call
    print("      A2E_MABU.py  Solve emitted for nofreq = %d frequencies" % nofreq)
    print(" UM = ", um2f(ofreq))
    t0       =  time.time()
    if (USE_MMAP):
        np.asarray([CELLS, nofreq], np.int32).tofile(f_emitted)
        EMITTED      =  np.memmap(f_emitted, dtype='float32', mode='r+', offset=8, shape=(CELLS,nofreq))
        EMITTED[:,:] = 0.0
    else:
        EMITTED      = np.zeros((CELLS, nofreq), np.float32)
    PEMITTED = []    # remains[] for dusts that are not mentioned in AALG = ususally most dust components
    if (AALG!=None): # @POL  we save also polarised emission to a separate file
        if (dust in AALG.keys()):
            print("=== POL ===  SolveEquilibriumDust added PEMITTED  for dust %s" % dust)
            if (USE_MMAP):
                np.asarray([CELLS, nofreq], np.int32).tofile(f_emitted)
                PEMITTED      =  np.memmap(f_emitted+'.P', dtype='float32', mode='r+', offset=8, shape=(CELLS,nofreq))
                PEMITTED[:,:] = 0.0
            else:
                PEMITTED      = np.zeros((CELLS, nofreq), np.float32)
    else:
        print("=== POL ===  SolveEquilibriumDust has no PEMITTED  for dust %s" % dust)

                
    if (len(MAP_UM)==1): 
        em_ifreq = argmin(abs(um2f(MAP_UM[0])-FREQ))
        FP_EBS = open('%s.eq_emission' % dust, 'wb')
    else:
        FP_EBS = None
        
        
    # **** BAD ****  ----   update has outer loop over FREQ,  inner over CELLS
    #                storage order has outer loop over CELLS, inner over FREQ 
    for ifreq in MOUT[0]:            # ifreq = selected frequencies, index to full list of frequencies
        oifreq = ifreq-MOUT[0][0]    # index to the set of output frequencies
        kernel_emission(commands, [GLOBAL,], [LOCAL,], FREQ[ifreq], KABS[ifreq], T_buf, EMIT_buf)
        cl.enqueue_copy(commands, TNEW, EMIT_buf)
        commands.finish()
        EMITTED[:, oifreq] = TNEW    # OUT OF ORDER UPDATE --- HOPE FOR SMALL nofreq!
        if (FP_EBS):
            if (ifreq==em_ifreq):  TNEW.tofile(FP_EBS)
        ## print("ifreq %3d   ofreq %3d   %10.3e" % (ifreq, oifreq, mean(TNEW)))
        if (len(PEMITTED)>0):        # polarised intensity for the current dust component
            # Since we have only one temperature for all Si grains, we can use a file
            # that gives R as a fraction of cross section in grains a>aalg.
            # Note that "tmp.rpol" is ratio where the denominator includes ALL dust species.
            # Here we must use  <Si_dust_name>.rpol where denominator is the total *Si* cross section.
            # See make_dust.py  + "cp tmp_aSilx.rpol simple_aSilx.rpol" !!!
            d    = loadtxt('%s.rpol' % (dust.replace('.dust', '')))
            Rpol = d[1:,1:]   # R[a, freq]
            apol = d[1:,0]    # minimum aligned grain size
            fpol = d[0,1:]    # frequency
            # interpolate to current frequency
            i    =  argmin(abs(fpol-FREQ[ifreq]))    #  correct column
            if (fpol[i]>FREQ[ifreq]): i = max([i-1, 0])
            j    =  min([i+1, len(fpol)-1])
            if (i==j): 
                wj = 0.0
            else:
                wj   =  (log(FREQ[ifreq])-log(fpol[i])) / (log(fpol[j])-log(fpol[i]))
            tmp  =  (1.0-wj)*Rpol[:,i] + wj*Rpol[:,j]  #  R(a) interpolated to correct frequency
            # interpolate in size, aalg -> R
            ipR  =  interp1d(apol, tmp, bounds_error=False, fill_value=0.0)
            aalg =  fromfile(AALG[dust], float32)[1:]   # a_alg for each cell, calculated by RAT.py
            PEMITTED[:, oifreq]  =   EMITTED[:, oifreq] * ipR(aalg)
    # **************************************************************************************
    print("      A2E_MABU.py  Solve emitted: %.2f seconds" % (time.time()-t0))
    if (FP_EBS): FP_EBS.close()
    
    if (USE_MMAP):
        del EMITTED
        if (AALG): del PEMITTED
    else:
        fp = open(f_emitted, 'wb')
        np.asarray([CELLS, nofreq], np.int32).tofile(fp)
        EMITTED.tofile(fp)  #  B(T)*kappa for a single dust component
        fp.close()
        del EMITTED
        if (len(PEMITTED)>0):
            fp = open(f_emitted+'.P', 'wb')   #  /dev/shm/tmp.emitted.P
            np.asarray([CELLS, nofreq], np.int32).tofile(fp)
            PEMITTED.tofile(fp)  #  B(T)*kappa for a single dust component
            fp.close()
            del PEMITTED
    return nofreq
        
        


FPE  = []
FPEP = None
    
fplog.write("\nAbsorption file: CELLS %d, NFREQ %d, NDUST %d\n" % (CELLS, NFREQ, NDUST))


# Read abundance files... we must have enough memory for that
ABU = np.ones((CELLS, NDUST), np.float32)
for idust in range(NDUST):
    if (len(AFILE[idust])>1): # we have a file for the abundance of the current dust species
        ABU[:,idust] = np.fromfile(AFILE[idust], np.float32, CELLS)
#for idust in range(NDUST):
#    print("IDUST=%d  ABUNDANCE %.3f" % (idust, np.mean(ABU[:,idust])))

# Initialise OpenCL to split the absorptions
# First check the number of frequencies in the absorption file...
fp_absorbed   = open(sys.argv[2], 'rb')
CELLS, NFFREQ = np.fromfile(fp_absorbed, np.int32, 2) 
fp_absorbed.close()
nkfreq        = NFFREQ


# at this point nkfreq is the number of frequencies passed to the split kernel
if (0):
    print("Initialize OpenCL for splitting absorptions => ALWAYS ON CPU !!")
    context, queue, mf = opencl_init(GPU=0, platforms=platforms)
else:
    print("Initialize OpenCL for splitting absorptions => GPU=%d" % GPU)
    context, queue, mf = opencl_init(GPU, platforms=platforms)
source      =  open(INSTALL_DIR+"/kernel_A2E_MABU_aux.c").read()
OPTS        =  '-D NFREQ=%d -D NDUST=%d' % (nkfreq, NDUST) 
program     =  cl.Program(context, source).build(OPTS)
Split       =  program.split_absorbed
Split.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None, None])
BATCH       =  32768
GLOBAL, LOCAL = BATCH, 16
ABS_IN_buf  =  cl.Buffer(context, mf.READ_ONLY,  4*BATCH*nkfreq)
ABS_OUT_buf =  cl.Buffer(context, mf.WRITE_ONLY, 4*BATCH*nkfreq)
RABS_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=RABS) # RABS[ NFREQ, NDUST]
ABU_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*BATCH*NDUST)                  # ABU[CELLS, NDUST]

# print("len(RABS)=%d,  nkfreq=%d" % (len(RABS), nkfreq))
# assert(len(RABS)==nkfreq)  # ????????
# len(RABS)=5,  nkfreq=250  ?????????


# loop over dust populations
### NOFREQ = NFREQ    # normally output is all simulated frequencies
fplog.write("\nLoop over dust components\n")


    




if (CR_HEATING>2): # need n(H) for dust-gas coupling calculation => READ THE CLOUD
    fp = open(CLOUD, 'rb')
    NX, NY, NZ, LEVELS, CELLS = fromfile(fp, int32, 5)
    LCELLS = zeros(LEVELS, int32)
    OFF    = zeros(LEVELS, int32)
    DENS   = zeros(CELLS, float32)
    cells      = 0
    kdensity   = KDENSITY    # scaling requested in the ini file
    for level in range(LEVELS):    
        if (level>0):
            OFF[level] = OFF[level-1] + cells   # index to [CELLS] array, first on this level
        cells = fromfile(fp, int32, 1)[0]       # cells on this level of hierarchy
        if (cells<0):
            break                               # the lowest level already read
        LCELLS[level] = cells    
        tmp = fromfile(fp, float32, cells)
        if (kdensity!=1.0):                     # apply density scaling
            m  = nonzero(tmp>0.0)               # not a link
            tmp[m] = clip(kdensity*tmp[m], 1.0e-6, 1e20)
        DENS[(OFF[level]):(OFF[level]+cells)] = tmp
    fp.close()
    # print("\nDENSITY READ\n\n")
    
    
    
    
# ad hoc Tgas and |Tgas-Tdust| (just for order of magnitude estimates)    
ip_Tg = interp1d([-8.0,   0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0, 7.0, 8.0, 20],
                 [ 15.0, 15.0, 15.0, 15.0, 14.0, 12.0, 10.0, 7.0, 6.0, 6.0, 6.0])
ip_DT = interp1d([-8.0,   0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0, 7.0, 8.0, 20.0],
                 [ 5.0,   5.0,  5.0,  5.0,  5.0,  5.0,  3.0, 1.0, 0.0, 0.0, 0.0])

nnemitted = []


for IDUST in range(NDUST):
    
    # for NN solve, nkfreq << NFFREQ
    print("A2E_MABU SPLIT =>  %d/%d   %s   nkfreq = %d,  NFFREQ = %d\n" % (1+IDUST, NDUST, DUST[IDUST], nkfreq, NFFREQ))
    t0 = time.time()
    # write a new file for absorptions, the photons absorbed by this dust component
    fp1 = open(ASHAREDIR+'/tmp.absorbed', 'wb')
    # nkfreq is now whatever was in the input absorption file
    asarray([CELLS, nkfreq], int32).tofile(fp1)
    fp_absorbed  = open(sys.argv[2], 'rb')
    CELLS, NFFREQ = np.fromfile(fp_absorbed, np.int32, 2) 
    print("DUST %d/%d -- CELLS %d, NFFREQ %d" % (1+IDUST, NDUST, CELLS, NFFREQ))
    fplog.write("  DUST %d/%d, CELLS %d, NFREQ %d ... split absorbed\n" % (IDUST, NDUST, CELLS, NFFREQ))
    if (NFFREQ!=NFREQ):
        print("ABSORPTIONS IN THE FILE: NFFREQ=%d NOT NFREQ=%d" % (NFFREQ, NFREQ))
        if (len(nnsolve)<1):
            for j in range(20):
                print("**** A2E_MABU.py .... EXIT !!!! ****")            
            sys.exit()  # in case of NN solution, NFFREQ << NFREQ
    # OpenCL used to split the absorptions => absorptions by the current dust species, 
    #    absorbed[icell] =  absorbed[icell] * RABS[ifreq,idust] / K
    #    K               =  sum[   ABU[icell, idust] * RABS[ifreq, idust]  ]
    #  process BATCH cells at a time
    
    # for CR_HEATING=2,3 we put the CR_HEATING or gas-dust coupling to last element of absorbed
    #  => kernel splits also that between the dust species
    #     ... the proportion is the ratio of absorption cross sections at the ***highest*** frequency
    #     ... which should be quite close to the ratio of physical cross sections = probability of collisions
    
    tmp_F       =  zeros((BATCH, NFFREQ), float32)  #  allocation to read the file
    tmp         =  zeros((BATCH, nkfreq), float32)  #  whatever kernel gives, NFREQ
    a = 0
    while(a<CELLS):
        b                 =  min(CELLS, a+BATCH)                             # cells [a,b[
        tmp_F[0:(b-a),:]  =  fromfile(fp_absorbed, np.float32, (b-a)*NFFREQ).reshape((b-a), NFFREQ)   # tmp[icell, ifreq]
        
        #  CR heating per H atom = 1e-27 erg/s  (i.e. for n(H)==1)  --- density information not needed !!
        #  coupling is 2.0e-33 * n(H2)**2 * (Tgas-Tdust) * (Tdust/10K)**0.5
        #    ~         2.0e-33 * 4 * n(H)**2  *  1K   *   (10K/10K)
        #              8.0e-33 *  n(H)**2
        # Young et al. (2004)
        #     coupling  1e-27 n(H2) (zeta/3e-17) (Delta Q/20eV)  erg/cm3/s
        #  ...  Lambda_gd = 9e-34   * n(H)^2 * sqrt(Tk) *  (1-0.8*exp(-75/Tk)) * (Tk-Td) * (Sigma_dust/6.09e-22)
        #                 = 9e-34   * n(H)^2 *  3       *     1                *   1K    *   1
        #                 = 2.7e-33 * n(H)^2
        #  upper limit  rate =  5.0e-33 * n(H)**2
        #      Or    Tgas  =  12K for n<=1e4,  10K for n=1e5, 7K for n>=1e6
        #            Tg-Td =   5K for n<=1e4,   3K for n=1e5, 1K for n>=1e6
        if   (CR_HEATING==1):       # this is upper limit -- all CR heating balanced by dust emission !!! 
            tmp_F[0:(b-a), NFREQ-1] = 1.0e-27*1.0e20  ; 
        elif (CR_HEATING==2):       # this is upper limit -- all CR heating balanced by dust emission !!! 
            tmp_F[0:(b-a), NFREQ-1] = 1.0e-27*1.0e20  * 2.0 ;  # HIGHER THAN NORMAL !!
        elif (CR_HEATING==3):       # this from gas-dust coupling with ad hoc Tg, Tg-Td
            # rate ~ RHO^2,   rate/H ~ RHO !!--v
            tmp_F[0:(b-a), NFREQ-1] = 9.0e-34 * DENS[a:b] *sqrt(ip_Tg(log10(DENS[a:b]))) * ip_DT(log10(DENS[a:b])) * 1.0e20
            
        if (nkfreq==NFFREQ):    # "normal", all read frequencies passed to the kernel
            cl.enqueue_copy(queue, ABS_IN_buf, tmp_F)                        # ABS_IN[batch, nkfreq], nkfreq=NFREQ
        else:                  
            # was USELIB... but that has been removed
            for j in range(20):
                print("*** ERROR in A2E_MABU.py: nkfreq!=NFREQ,  %d != %d ???" % (nkfreq, NFREQ))
            sys.exit()
        # absorbed energy split in proportion to absorption cross sections x abundance
        cl.enqueue_copy(queue, ABU_buf, ABU[a:b, :])                   # ABU[batch, ndust]
        # for CR_HEATING>0,  last NFREQ-1 element is additional heating, ignored in the normal frequency integration
        Split(queue, [GLOBAL,], [LOCAL,], IDUST, b-a, RABS_buf, ABU_buf, ABS_IN_buf, ABS_OUT_buf)
        cl.enqueue_copy(queue, tmp[0:(b-a), :], ABS_OUT_buf)           # tmp[BATCH, nkfreq], emission for b-a cells
        tmp[0:(b-a),:].tofile(fp1)                                     # absorption file possibly only reference frequencies
        a             +=  BATCH
        # Split   ==>    absorbed_i  =  absorbed *  k_i / (sum(k_j * abu_j))
        #     energy absorbed by dust i, divided by abu_i 
        #     == energy absorbed by grain, normalised for abu=1.0 case
    # --------------------------------------------------------------------------------
    fp_absorbed.close()
    fp1.close() # 8 + 4*CELLS*NFREQ bytes    ASHAREDIR+'/tmp.absorbed'
    
    # print("=== Split absorbed: %.2f seconds" % (time.time()-t0))
    fplog.write("      Split absorbed: %.2f seconds\n" % (time.time()-t0))

    fplog.write("      Solve emission, NOFREQ %d, NFREQ %d, nkfreq %d\n" % (NOFREQ, NFREQ, nkfreq))
    #print("NOFREQ = NFREQ = %d .... dust %s" % (NOFREQ, DUST[IDUST]))
    #print("")


    
    if (0):
        # AT THIS POINT ABSORBED IS CORRECT == IDENTICAL BETWEEN NORMAL AND nnsolve RUNS
        # BEFORE ANY FF CORRECTIONS => NO FF CORRECTIONS TO BE APPLIED ????
        if (len(nnsolve)<1): # normal run
            os.system('cp tmp.absorbed  ABS_normal_%s.dump' % DUST[IDUST])
        else:
            os.system('cp tmp.absorbed  ABS_nnsolve_%s.dump' % DUST[IDUST])

            
    t0 = time.time()
    if (len(nnsolve)>0):  
        # we solve emission using a NN fit,  nnabs wavelengths mapped to nnemit wavelengths
        # we combine these in memory and therefore also skip the rest of the loop
        # NNSolve returns solution for current dust, all cells, all frequencies (=nnemit)
        #    tmp.absorbed  ==    k_i / sum(k_j*abu_j)  *   absorbed
        #    tmp.emitted   ==    emission per unit density, unit abundance

        # print("A2E_MABU.py:667 calling NN_solve")
        t0000 = time.time()
        Na = len(nnabs)
        Ne = len(nnemit)
         
        if (0): # @@@
            ifreq2 = argmin(abs(nnabs-um2f(2.0)))
            tmp    = fromfile(ESHAREDIR+'/tmp.absorbed', float32)[2:].reshape(CELLS, Na)[:, ifreq2]
            asarray(tmp, float32).tofile('NN_ABS_%s.dump' % DUST[IDUST])
            
            
        NN_solve(nnsolve, DUST[IDUST], nnabs, nnemit, ASHAREDIR+'/tmp.absorbed', ESHAREDIR+'/tmp.emitted', nngpu=nngpu)

        
        if (0): # @@@
            ifreq100 = argmin(abs(nnemit-um2f(250.0)))
            tmp      = fromfile(ESHAREDIR+'/tmp.emitted',  float32)[2:].reshape(CELLS, Ne)[:, ifreq100]
            asarray(tmp, float32).tofile('NN_%s.dump' % DUST[IDUST])
            
            
                
                
        if (1):
            if (len(nnemitted)<1): nnemitted  = np.multiply(fromfile(ESHAREDIR+'/tmp.emitted', float32)[2:].reshape(CELLS, Ne), ABU[:, IDUST:(IDUST+1)])
            else:                  nnemitted += np.multiply(fromfile(ESHAREDIR+'/tmp.emitted', float32)[2:].reshape(CELLS, Ne), ABU[:, IDUST:(IDUST+1)])
        else:
            if (len(nnemitted)<1): 
                nnemitted  =  zeros((CELLS, Ne), float32)
            # read emission for the current dust, still without abundance scaling
            tmp        =  fromfile(ESHAREDIR+'/tmp.emitted', float32)[2:].reshape(CELLS, Ne)
            for j in range(Ne): # loop over emitted frequencies, scale now with the abundance of this dust
                nnemitted[:, j]  +=   tmp[:,j] * ABU[:,IDUST]
            
                
        if (IDUST==(NDUST-1)):  # last dust => write the file and exit the loop (to close fplog and to exit the program)
            fp3 = open(sys.argv[3], 'wb')
            asarray([CELLS, Ne], int32).tofile(fp3)
            nnemitted.tofile(fp3)  #  nnemitted[CELLS, Ne]
            fp3.close()
            del nnemitted
            os.system('ls -l %s' % sys.argv[3])
            break
        # print(" A2E_MABU.py:667 calling NN_solve: %.3f\n\n" % (time.time()-t0000))
        assert(FPE==[])
        continue  # continue the loop with the next dust
    
        
        
    if (EQDUST[IDUST]):
        # Equilibrium dust, also calculating emission to SHAREDIR/tmp.emitted
        # MAY INCLUDE ONLY FREQUENCIES [UM_MIN, UM_MAX]
        # Note on CR_HEATING
        #   CR_HEATING=1  
        #                 only if there is a single equilibrium dust component,
        #                 assumes full coupling between gas and dust => upper limit for CR heating!
        #   CR_HEATING=2 
        #                 CR heating takes into account density-dependent coupling
        #                 heating is split between the dust species in proportion of kabs
        #                 in practice, the CR heating rate is transmitted to kernel in the last frequency channel!
        print("=== SolveEquilibriumDust(%s) === ..... GPU = " % DUST[IDUST], GPU)
        nofreq = SolveEquilibriumDust(DUST[IDUST], ASHAREDIR+'/tmp.absorbed', ESHAREDIR+'/tmp.emitted', 
        UM_MIN, UM_MAX, MAP_UM, GPU, platforms, AALG)
        fplog.write("      SolveEquilibriumDust(%s) => nofreq %d, NOFREQ %d\n" % (DUST[IDUST], nofreq, NOFREQ))
    else:        
        # note -- GPU can be  e.g. 0.1 for CPU/device=1  or 1.3 for GPU/device=3
        # 2021-05-03  ---  if MAP_UM is given, use the first frequency ==> A2E.py will still solve emission
        #                  for either all the frequencies or just for a single frequency
        em_ifreq = -1    # A2E.py will solve all frequencies or only one frequency !
        nstoch   = 999   # currently all bins as stochastically heated
        ## nstoch   = 0
        if (len(MAP_UM)>0): 
            em_ifreq = argmin(abs(um2f(MAP_UM[0])-FREQ))
        else:
            # we may also try UM_MIN, UM_MAX --- if these select a single frequency, use that
            # otherwise emission will be calculated for all frequencies
            mm  =  nonzero((f2um(FREQ)>=UM_MIN)&(f2um(FREQ)<=UM_MAX))
            if (len(mm[0])==1):  em_ifreq = m[0]  # select a single output frequency
        if (AALG==None):
            
            
            if (0): # @@@
                ifreq2 = argmin(abs(FREQ-um2f(2.0)))
                tmp = fromfile(ESHAREDIR+'/tmp.absorbed', float32)[2:].reshape(CELLS, NFREQ)[:, ifreq2]
                asarray(tmp, float32).tofile('A2E_ABS_%s.dump' % DUST[IDUST])

                
            print("=1========================================================================================")
            print('      A2E.py  %s.solver %s/tmp.absorbed %s/tmp.emitted  %.1f %d %d'         % (DUST[IDUST], ASHAREDIR, ESHAREDIR, GPU_TAG, nstoch, em_ifreq))
            print("==========================================================================================")
            fplog.write('      A2E.py  %s.solver %s/tmp.absorbed %s/tmp.emitted  %.1f %d %d\n' % (DUST[IDUST], ASHAREDIR, ESHAREDIR, GPU_TAG, nstoch, em_ifreq))
            os.system('A2E.py  %s.solver %s/tmp.absorbed %s/tmp.emitted  %.1f %d  %d'          % (DUST[IDUST], ASHAREDIR, ESHAREDIR, GPU_TAG, nstoch, em_ifreq))
            
            
            if (0): # @@@
                ifreq100 = argmin(abs(FREQ-um2f(250.0)))
                tmp = fromfile(ESHAREDIR+'/tmp.emitted', float32)[2:].reshape(CELLS, NFREQ)[:, ifreq100]
                asarray(tmp, float32).tofile('A2E_%s.dump' % DUST[IDUST])
            
        else:
            # Solve emission and separately the polarised emission using the aalg file and the file  <dust>.rpol_single
            if (DUST[IDUST] in AALG.keys()):  # this dust will have polarised intensity
                aalg_file  =  AALG[DUST[IDUST]]
                print("=2========================================================================================")
                print('      A2E.py  %s.solver %s/tmp.absorbed %s/tmp.emitted  %.1f %d %d %s'         % (DUST[IDUST], ASHAREDIR, ESHAREDIR, GPU_TAG, nstoch, em_ifreq, aalg_file))
                print("==========================================================================================")
                fplog.write('      A2E.py  %s.solver %s/tmp.absorbed %s/tmp.emitted  %.1f %d %d %s\n' % (DUST[IDUST], ASHAREDIR, ESHAREDIR, GPU_TAG, nstoch, em_ifreq, aalg_file))
                # aalg_file = a_alg[CELLS], minumum aligned grain size for each cell
                #                  solver    absorbed        emitted         GPU_TAG   nstoch  IFREQ  aalg
                os.system('A2E.py  %s.solver %s/tmp.absorbed %s/tmp.emitted  %.1f  %d      %d     %s' % (DUST[IDUST], ASHAREDIR, ESHAREDIR, GPU_TAG, nstoch, em_ifreq, aalg_file))
                # we have emission in %s/tmp.emitted, polarised emission %s/tmp.emitted.P
            else:  # without polarised emission
                print("=3========================================================================================")
                print('      A2E.py  %s.solver %s/tmp.absorbed %s/tmp.emitted  %.1f %d %d'         % (DUST[IDUST], ASHAREDIR, ESHAREDIR, GPU_TAG, nstoch, em_ifreq))
                print("==========================================================================================")
                fplog.write('      A2E.py  %s.solver %s/tmp.absorbed %s/tmp.emitted  %.1f %d %d\n' % (DUST[IDUST], ASHAREDIR, ESHAREDIR, GPU_TAG, nstoch, em_ifreq))
                os.system('A2E.py  %s.solver %s/tmp.absorbed %s/tmp.emitted  %.1f %d %d'           % (DUST[IDUST], ASHAREDIR, ESHAREDIR, GPU_TAG, nstoch, em_ifreq))

                
        #if (DUST[IDUST]=='CMCa'):
        #    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        #    time.sleep(30)
        #    sys.exit()

        if (NOFREQ!=NFREQ):  # so far only equilibrium dust files can have NOFREQ<NFREQ !!
            print("=== A2E_MABU.py --- stochastically heated grains with equilibrium grains")
            print("    different number of frequencies... fix A2E.py to use a2e_wavelength parameter??")
            if (NOFREQ!=1):
                sys.exit()
            else:
                print("... EXCEPT IT IS OK TO HAVE JUST A SINGLE OUTPUT FREQUENCY !!")

                

    if (len(nnmake)>0): 
        # we have calculated mapping ASHAREDIR+'/tmp.absorbed' ->  ESHAREDIR+'/tmp.emitted' for all frequencies
        # use that to create a NN mapping for the current dust, DUST[IDUST]
        # Call NN_fit with only the nnabs frequencies, do not overwrite the all-frequency file tmp.emitted
        # tmp_absorbed[cells, nfreq]  ->  tmp_absorbed[cells, nnabs]
        
        # absorbed.data has been scaled with FF = integration weight that DEPENDS ON THE FREQUENCY GRID
        # this is different for the full frequency grid here and later when NN is used to solve emission
        # based on a smaller number of frequencies
        # ==>  divide absorbed by the current FF, both when making the fit and later when 
        #      absorbed are calculated with fewer frequencies
        #      include ad hoc scaling by 1e20 to have absorptions close to unity order
        NFREQ = len(FREQ)
        nfreq = len(nnabs)
        #FF_now = zeros(len(FREQ), float32)
        #for ifreq in range(NFREQ):
        #    FF_now[ifreq]  =  FREQ[ifreq] / 1.0e20   # AD HOC 
        #    if (ifreq==0):              FF_now[ifreq] *= 0.5*(FREQ[1]-FREQ[0])
        #    else:
        #        if (ifreq==(NFREQ-1)):  FF_now[ifreq] *= 0.5*(FREQ[NFREQ-1]-FREQ[NFREQ-2])
        #        else:                   FF_now[ifreq] *= 0.5*(FREQ[ifreq+1]-FREQ[ifreq-1])
        #FF_now = FF_now[IND_nnabs]  # current FF, for the nnabs frequencies only
        # print()
        # scaled absorptions for frequencies in nnabs
        cells, nf= fromfile(ASHAREDIR+'/tmp.absorbed', int32, 2)
        assert(nf==NFREQ)  # must be the full frequency grid
        # extract absorptions only for the frequencies in nnabs
        tmp          = fromfile(ASHAREDIR+'/tmp.absorbed', float32)[2:].reshape(cells, nf)[:, IND_nnabs]
        if (0):
            for ifreq in range(nfreq):
                print("#@ make  ifreq=%d  raw abs %12.4e" % (ifreq, np.mean(tmp[:,ifreq])))
        ###
        #if (0): # NORMALISATION WITH FF WAS NOT NEEDED AFTER ALL ????
        #    for ifreq in range(nfreq):          # remove FF by dividing by it
        #        tmp[:,ifreq] /= FF_now[ifreq]   # ... including ad hoc 1e20 scaling for absorptions going into NN
        #        print("#@ make  ifreq=%d, %.3e Hz,  FF %12.4e  normalised absorptions <> = %12.4e" % (ifreq, nnabs[ifreq], FF_now[ifreq], np.mean(tmp[:,ifreq])))
        with open(ASHAREDIR+'/nn.absorbed', 'wb') as fp:
            asarray([cells, len(nnabs)], int32).tofile(fp)
            asarray(tmp, float32).tofile(fp)
        # emissions for frequencies in nnemit
        cells, nfreq = fromfile(ASHAREDIR+'/tmp.emitted', int32, 2)
        tmp          = fromfile(ASHAREDIR+'/tmp.emitted', float32)[2:].reshape(cells, nfreq)[:, IND_nnemit]
        with open(ASHAREDIR+'/nn.emitted', 'wb') as fp:
            asarray([cells, len(nnemit)], int32).tofile(fp)
            asarray(tmp, float32).tofile(fp)
        ###    training based on  nn.absorbed -> nn.emitted
        del tmp
        if (0):
            print("cells %d, nfreq %d, nnabs %d, nnemit %d" % (cells, nfreq, len(nnabs), len(nnemit)))
            print("IND_nnabs ",  IND_nnabs)
            print("        ", f2um(FREQ[IND_nnabs]))
            print("IND_nnemit ", IND_nnemit)
            print("        ", f2um(FREQ[IND_nnemit]))
        #  NN fitting  absorbed*1e20/FF <--> emitted
        NN_fit(nnmake, DUST[IDUST], CELLS, nnabs, nnemit, ASHAREDIR+'/nn.absorbed', ESHAREDIR+'/nn.emitted', nngpu=nngpu)
        # continue as without NN, tmp.emitted still has the emission for all frequencies
        
        
    # read the emissions and add to FPE
    if (IDUST==0): # FPE opened only here, once we know the number of output frequencies
        if (USE_MMAP):
            fp       =  open(sys.argv[3], 'wb')
            np.asarray([CELLS, NOFREQ], np.int32).tofile(fp)
            fp.close()
            FPE      =  np.memmap(sys.argv[3], dtype='float32', mode='r+', shape=(CELLS, NOFREQ), offset=8)
            FPE[:,:] =  0.0
        else:
            FPE      =  np.zeros((CELLS, NOFREQ), np.float32)
        # polarisation
        if (AALG!=None):
            if (USE_MMAP):
                fp        =  open(sys.argv[3]+'.R', 'wb')    # polarisation reduction, initial sum of polarised intensity
                np.asarray([CELLS, NOFREQ], np.int32).tofile(fp)
                fp.close()
                FPEP      =  np.memmap(sys.argv[3]+'.R', dtype='float32', mode='r+', shape=(CELLS, NOFREQ), offset=8)
                FPEP[:,:] =  0.0
            else:
                FPEP      =   np.zeros((CELLS, NOFREQ), np.float32)
            
            
            
    # print("=== Add emitted to sum file: %s  DUST %s" % (sys.argv[3], DUST[IDUST]))
    t0 = time.time()

    ## sys.exit()
    
    filename         =  ESHAREDIR+'/tmp.emitted'        
        
    fp2              =  open(filename, 'rb')
    cells_e, nfreq_e =  np.fromfile(fp2, np.int32, 2)  # get rid of the header (CELLS, NFREQ)
    print("    emitted file for %s, cells %d, CELLS %d, nfreq %d, NOFREQ %d" % (DUST[IDUST], cells_e, CELLS, nfreq_e, NOFREQ))
    fplog.write('      Add emission from %s .... processing %s with NOFREQ=%d\n' % (filename, DUST[IDUST], NOFREQ))    

    fp2P = None
    if (AALG!=None):
        print('AALG ', AALG)
        if (DUST[IDUST].replace('.dust','') in AALG.keys()):
            print("Open %s.P for polarised emission from dust %s" % (filename, DUST[IDUST]))
            fp2P             =  open(filename+'.P', 'rb')
            cells_e, nfreq_e =  np.fromfile(fp2P, np.int32, 2)  # get rid of the header (CELLS, NFREQ)
        else:
            print("Dust %s will not have polarised emission" % DUST[IDUST])

    t00 = time.time()
    if (1):
        # SLOW !
        for ICELL in range(CELLS):
            # file = B(T)*KABS for a single dust component, total is sum of B(T)*KABS*ABU
            # FPE[ICELL,:] += np.fromfile(fp2, np.float32, NOFREQ) * ABU[ICELL, IDUST]
            xxxx = np.fromfile(fp2, np.float32, NOFREQ)
            if (ICELL%1000000==0): 
                print("=== A2E_MABU %8d/%8d, add emission, NOFREQ %d" % (ICELL, CELLS, len(xxxx)))
            FPE[ICELL,:] += xxxx * ABU[ICELL, IDUST]
    else:
        # MUCH FASTER
        a = 0
        while(a<CELLS):
            b = min(a+1024, CELLS)
            if (0):
                print()
                print(" ......... CELLS %d - %d OUT OF %d, NOFREQ=%d" % (a, b, CELLS, NOFREQ))
                print(" ......... FPE[a:b,:]",      FPE[a:b,:].shape)
                print(" ......... (b-a, NOFREQ) ",  b-a, NOFREQ)
                print(" ......... ABU[a:b, IDUST]", ABU[a:b, IDUST].shape)
            FPE[a:b,:] += np.fromfile(fp2, np.float32, (b-a)*NOFREQ).reshape(b-a, NOFREQ) * \
                          ABU[a:b, IDUST].reshape(b-a,1)
            a += 1024
        # polarisation
        if (fp2P):
            print("=== POL ===  Add polarised emission from %s into FPEP" % DUST[IDUST])
            a = 0
            while(a<CELLS):
                b = min(a+1024, CELLS)
                FPEP[a:b,:] += np.fromfile(fp2P, np.float32, (b-a)*NOFREQ).reshape(b-a, NOFREQ) * \
                               ABU[a:b, IDUST].reshape(b-a,1)
                a += 1024
        else:
            print("=== POL ===  No polarised emission added to FPEP from %s" % DUST[IDUST])
            
    fplog.write('      Added emitted to FPE: %.2f seconds\n' % (time.time()-t00))
    
    fp2.close()
    if (fp2P): 
        fp2P.close()
    print("    Add emitted to sum file: %.2f seconds =====" % (time.time()-t0))
    # os.system('rm /dev/shm/tmp.absorbed /dev/shm/tmp.emitted')

    if (KEEP_COMPONENT_EMITTED):
        os.system('mv -f %s  tmp.emitted.DUST_%02d' % (filename, IDUST))
    

print("DONE !!!")
os.system('ls -l emitted.data')

        

fplog.write('Loop over dusts completed\n')
fplog.write('\n[%s]\n' % time.asctime())
fplog.close()

if (len(nnsolve)>0): sys.exit()   # emission file was already saved

if (USE_MMAP):
    if (AALG):
        # before FPE and FPEP are destroyed, convert FPEP from polarised intensity to 
        # polarisation reduction factor
        FPEP[:,:] /= FPE[:,:]
    del FPE  # file should now contain combined emission from all dust components
    if (AALG): del FPEP
else:
    if (AALG):  FPEP /= (FPE+1.0e-32)   # polarised intensity -> polarisation reduction factor
    fp  =  open(sys.argv[3], 'wb')
    np.asarray([CELLS, NOFREQ], np.int32).tofile(fp)
    if (len(FPE)>0): FPE.tofile(fp)   #  for nnsolve FPE==[]
    fp.close()
    del FPE
    if (AALG):
        fp  =  open(sys.argv[3]+'.R', 'wb')
        #  A2E_MABU.py writes R for all NOFREQ but currently ASOC uses only a single R vector
        #  => polarisation maps will be done with ASOC.py one map per mapping run.
        # However, here we may save in one A2E_MABU.py run R for several frequencies.
        # One must then copy the R vector of the correct frequency to a separate file before ASOC.py run
        # For simplicity, one may just limit NOFREQ to one...
        # For compatibility with RAT.py, the R-file header is only {CELLS} (not {CELLS, NOFREQ}!)
        np.asarray([CELLS,], np.int32).tofile(fp)   
        FPEP.tofile(fp)
        fp.close()
        del FPEP

        
os.system('cp emitted.data normal.emitted.data')

