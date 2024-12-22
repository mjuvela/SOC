#!/usr/bin/env python
import time

if (0):
    from MJ.mjDefs import *
    from MJ.Aux.DustLib import *
else:
    import os,sys
    import numpy as np

"""
Usage:
    ASOC_driver.py  soc-ini  [uselib] [makelib] 
        
If calculations involve stochastically heated grains:
    - (1) creates <dust>.solver files with A2E_pre.py
    - (2) creates <dust>_simple.dust files
    - (3) runs ASOC.py with <dust>_simple.dust files
    - (4) solves dust emission with A2E_MABU.py
    - (5) calculates maps with a second ASOC.py run


If calculation involve only eqdust (no stochastically heated ones),
do also the above three runs -- this is needed in case of 
spatially varying dust abundances.   

2019-09-22: 
    For the USELIB case, if ini-file contains the keyword ofreq + filename,
    the emitted file will contain only the frequencies listed in the file;
    ini will be read only in A2E_MABU and passed on as command line argument to A2E_LIB.
    
    Simulation with fewer frequencies is done using the normal
    <dust>_simple.dust but with the added keyword fselect.
    
2019-11-20:
    accept non-simple dust files without the initial gs_

2023-12-30:
    dropped old LIB in favour of neural networks
    
    nnmake         ==>     all frequencies in asborbed.data and emitted.data
                           while NN is trained, output emitted.data still constains all frequencies
    nnmake + thin  ==>     ASOC_driver calls A2E_MABU with absorbed[CELLS::nnthin, :],
                           emitted.data will be incomplete = contain only some cells
                           => map making can proceed but for nnemit frequencies only
    nnsolve        ==>     RT and maps using nnabs/nnemit frequencies only        
"""

if (len(sys.argv)<2):
    print("Usage:  ASOC_driver.py  soc.ini \n")
    sys.exit()

    
FREQ  = np.loadtxt('freq.dat')    # ASSUME THAT THIS IS ALWAYS AVAILABLE?
NFREQ = len(FREQ)


# Read the original ini file -- one should also make sure "noabsorbed" is dropped, it is now user's responsibility 
INI   = sys.argv[1]
LINES = open(INI).readlines()


def is_gs_dust(name):
    if (open(name).readline().split()[0]=='gsetdust'): return True
    return False

def get_floats(s):
    # for list of strings, return maximum number of floats that could be read
    res = []
    for i in range(len(s)):
        print(s[i])
        try:
            x = float(s[i])
            res.append(x)
        except:
            pass
    return np.asarray(res, np.float32)


def um2f(um):
    return 2.997924580e14/um

def f2um(freq):
    return 2.997924580e14/freq


# for nnmake + nnthin,  ASOC_driver will call A2E_MABU with a smaller absorbed files
nnabs, nnemit, nnmake, nnsolve, nnthin = [], [], '', '', 1
nenumber = 256

# Make a list of the dusts ... AND ABUNDANCE FILES, IF GIVEN
DUST, ABUNDANCE, STOCHASTIC = [], [], []
fabs, femit = None, None      # names of absorption and emission files
for line in LINES:
    s = line.split()
    if (len(s)<2): continue
    if (s[0][0:1]=='#'): continue    
    if (s[0]=='optical'):
        name = s[1]
        DUST.append(name.replace('.dust',''))
        if (is_gs_dust(name)):  # this could be "gs_aSilx"
            STOCHASTIC.append(1)
        else:                   # this coul de "aSilx_simple"
            STOCHASTIC.append(0)
        abu  = ''
        if (len(s)>2):
            if (s[2]!='#'):
                abu = s[2]
        ABUNDANCE.append(abu)
    if (s[0]=='absorbed'): fabs    = s[1]
    if (s[0]=='emitted'):  femit   = s[1]
    # NN options...  nnsolve, nnabs, nnemit
    if (s[0].find('nnabs')>=0):
        nnabs  =  get_floats(s[1:])    # wavelengths included in absorptions (NN mapping)
        nnabs  =  np.sort(um2f(nnabs)) #   to frequencies in increasing order
    if (s[0].find('nnemit')>=0):
        nnemit =  get_floats(s[1:])    # wavelengths included in emission (NN mapping)
        nnemit =  np.sort(um2f(nnemit))  #   to frequencies in increasing order
    if (s[0].find('nnmake')>=0):       # make => using full sets of input and output frequencies
        nnmake = s[1]                  #  prefix for  <prefix>_<dust>.nn 
    if (s[0].find('nnsolve')>=0):      # solve  =   nnabs -> nnemit frequencies only
        nnsolve = s[1]
    if (s[0].find('nnthin')>=0):       # A2E_driver has taken care of nnthin already
        nnthin = int(s[1])             # in absorbed.data
    if (s[0].find('nenumber')>=0):
        nenumber = int(s[1])   
     

# for NN runs, get indices into FREQ for nnabs and for nnemit
IND_nnabs, IND_nnemit = [], []
if (len(nnabs)>0):
    IND_nnabs = np.zeros(len(nnabs), np.int32)
    for i in range(len(nnabs)):
        f = nnabs[i]
        k = np.argmin(abs(f-FREQ))
        if ((abs(f-FREQ[k])/f)>0.02):
            print("*** Error in A2E_MABU: nnabs %.3f does not correspond to any frequency" % f2um(nnabs[i]))
            print("  nnabs %.3f um, closest %.3f um" % (f2um(nnabs[i]), f2um(FREQ[k])))
            sys.exit()
        IND_nnabs[i] = k  #    FREQ[IND_nnabs[i]] ~ nnabs[i]
if (len(nnemit)>0):
    IND_nnemit = np.zeros(len(nnemit), np.int32)
    for i in range(len(nnemit)):
        f = nnemit[i]
        k = np.argmin(abs(f-FREQ))
        if ((abs(f-FREQ[k])/f)>0.02):
            print("*** Error in A2E_MABU: nnemit %.3f um does not correspond to any frequency" % f2um(nnemit[i]))
            sys.exit()
        IND_nnemit[i] = k  #    FREQ[IND_nnemit[i]] ~ nnemit[i]


STOCHASTIC = np.asarray(STOCHASTIC, np.int32)
NDUST      = len(DUST)
    
print("================================================================================")
print("ASOC_driver")
for idust in range(NDUST):
    print("%30s, stochastic=%d, [%s]" % (DUST[idust], STOCHASTIC[idust], ABUNDANCE[idust]))



    
def write_nndust(dust, freq, tag):
    """
    Write a version of the simple dust, limited to frequencies freq. Add tag as suffix before '.dust'
    Dust file is assumed to contain four lines before the main array.
    """
    lines  = open(dust).readlines()
    nndust = dust.replace('.dust', '_%s.dust' % tag)
    fp     = open(nndust, 'w')
    fp.write(lines[0])
    fp.write(lines[1])
    fp.write(lines[2])
    fp.write('%d\n' % len(freq))
    d    = np.loadtxt(dust, skiprows=4)
    if (1):
        tmp = sum(((d[:,0]-FREQ)/d[:,0])**2)
        if (tmp>0.1):
            print("*** WARNING from write_nndust: freq.dat does not agree with  %s ??" %dust)
    for i in range(len(freq)):
        r = (abs(d[:,0]-freq[i]))/d[:,0]  # relative frequency difference
        k = np.argmin(r)                  # d[:,0][k] is the closest entry
        if (r[k]>0.03):
            print("*** WARNING -- write_nndust -- wavelength %.2f um, closest match %.2f um" %   (f2um(freq[i]), f2um(d[:,0][k])))
        fp.write("%12.5e %7.3f  %12.4e %12.4e  # %9.2f um\n" % (d[k,0], d[k,1], d[k,2], d[k,3], f2um(freq[i])))
    fp.close()
    return nndust
    


# Write solver file for each stochastically heated dust
for idust in range(NDUST):
    if (STOCHASTIC[idust]==0): continue
    dust    =  DUST[idust]
    solver  =  '%s.solver' % dust
    # A2E_MABU.py will drop the "gs_" prefix from stochastic => do the same here for the solver file
    if (solver[0:3]=='gs_'): solver = solver[3:] # A2E_MABU uses aSilx.solver, not gs_aSilx.solver
    redo    =   True
    if (os.path.exists(solver)): # skip solver creation if that exists and is more recent than dust file
        if (os.stat(dust+'.dust').st_mtime<os.stat(solver).st_mtime): redo = False
    if (redo):
        print("================================================================================")
        print('A2E_pre.py %s.dust freq.dat %s %d' % (dust, solver, nenumber)) # gs_aSilx.dust -> aSilx.solver
        print("================================================================================")
        t0 = time.time()
        os.system('A2E_pre.py %s.dust freq.dat %s %d' % (dust, solver, nenumber))
        # print('... A2E_pre.py %s.dust freq.dat %s ... %.2f seconds' % (dust, solver, time.time()-t0)) # gs_aSilx.dust -> aSilx.solver

        
        
fp    = open('rt_simple.ini', 'w')
idust = 0
for line in LINES:
    s = line.split()
    if (len(s)<1): continue
    if (s[0].find('noabsorbed')>=0): continue
    if (s[0].find('libabs')>=0):     continue   # in ASOC_driver.py, simulation and library for all NFREQ frequencies !!!
    if (s[0].find('libmaps')>=0):    continue
    if (s[0]!='optical'):
        fp.write(line)
for idust in range(NDUST):
    dust = DUST[idust]
    if (STOCHASTIC[idust]==0):  # simple dust already
        fp.write('optical %s.dust %s\n' % (dust, ABUNDANCE[idust]))
    else:
        # again prefix 'gs_' dropped from the name of simple dusts
        fp.write('optical %s_simple.dust %s\n' % (dust.replace('gs_', ''), ABUNDANCE[idust]))
fp.write('nomap\n')
fp.write('nosolve\n')

fp.close()


if (len(nnsolve)>0):
    # RT is done using only nnabs wavelengths
    # make alternative ini file with only with nnabs wavelengths for
    #   - dusts       
    #   - background               "background  some.bg  1.5 1"
    #   - pointsource luminosity   "pointsource 1.1 2.2 3.3  luminosity.bin 4.1"
    fp = open('rt_simple_nn.ini', 'w')
    for line in open('rt_simple.ini').readlines():
        s = line.split()
        if (len(s)<2):
            fp.write(line)
            continue
        if (s[0][0:1]=='#'): continue        
        if (s[0].find('optical')>=0):
            dust = s[1]
            nndust = write_nndust(dust, nnabs, 'nnabs')  # only nnabs frequencies
            abu    = ""
            if (len(s)>2):  abu  = s[2]    # could have ***abundance***
            fp.write('optical %s  %s\n' % (nndust, abu))
        elif (s[0].find('backgro')>=0):
            # s[1] if the intensity file => cut from nfreq to nnabs
            nnbg  = np.fromfile(s[1], np.float32)[IND_nnabs]
            np.asarray(nnbg, np.float32).tofile(s[1]+'.nn')
            for k in range(len(s)):
                if (k==1):  fp.write('%s.nn ' % s[k])
                else:       fp.write('%s '    % s[k])
            fp.write('\n')
        elif (s[0].find('pointso')>=0):
            # s[4] if the intensity file => cut from nfreq to nnabs
            nnps   = np.fromfile(s[4], np.float32)[IND_nnabs]
            np.asarray(nnps, np.float32).tofile(s[4]+'.nn')
            for k in range(len(s)):
                if (k==4):  fp.write('%s.nn ' % s[k])
                else:       fp.write('%s ' % s[k])
            fp.write('\n')
        elif (s[0].find('dsc')>=0): # scattering function
            bins = int(s[2])
            # CSC =   [NFREQ, BINS] + [NFREQ, BINS]
            # one with nnabs (for RT), one with nnemit frequencies (for map making)
            a, b = [], []
            with open(s[1], 'rb') as fpdsc:
                a = np.fromfile(fpdsc, np.float32, NFREQ*bins).reshape(NFREQ, bins)
                b = np.fromfile(fpdsc, np.float32, NFREQ*bins).reshape(NFREQ, bins)
            with open('nnabs.dsc', 'wb') as fpdsc:
                np.asarray(a[IND_nnabs,:], np.float32).tofile(fpdsc)
                np.asarray(b[IND_nnabs,:], np.float32).tofile(fpdsc)
            with open('nnemit.dsc', 'wb') as fpdsc:
                np.asarray(a[IND_nnemit,:], np.float32).tofile(fpdsc)
                np.asarray(b[IND_nnemit,:], np.float32).tofile(fpdsc)
            # at this point we are doinf RT => replace dsc file with nnabs.csc
            fp.write('dsc nnabs.dsc %s\n' % s[2])
        else:
            fp.write(line)
    fp.close()        
    print("================================================================================")
    print('ASOC.py rt_simple_nn.ini')
    print("================================================================================")
    t0 = time.time()
    os.system('ASOC.py rt_simple_nn.ini')
    print('@@ ASOC.py rt_simple_nn.ini ... %.2f seconds' % (time.time()-t0))
    
else: 
    
    # othwerwise normal ASOC run with all frequencies
    # in case of nnmake,  A2E_MABU will solve emission for all frequencies
    #                     and trains the NN => run can proceed normally to map making @@
    # but with nnmake+thin emitted.data contains only part of the cells => map making skipped
    
    print("================================================================================")
    print('ASOC.py rt_simple.ini')
    print("================================================================================")
    if (1):
        t0 = time.time()
        os.system('ASOC.py rt_simple.ini')
        print('@@ ASOC.py rt_simple.ini ... %.2f seconds' % (time.time()-t0))
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!! ASOC.py rt_simple.ini SKIPPED !!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        time.sleep(10)


# A2E_MABU.py run -- solve emission for each dust
args = ""
# calculate emission --- need <dust>.dust


# Give A2E_MABU always the original dusts, e.g. gs_aSilx.dust
# ... no !!!  if the init contains something like aSilx_simple.dust
# the user has chosen that and A2E_MABU.py will solve emission without
# stochastic heating for that dust component
lines   = open(INI).readlines()
if (0):
    fpo     = open('/dev/shm/a2e.ini', 'w')
else:
    fpo     = open('a2e.ini', 'w')
libmaps = ''
for line in lines:
    fpo.write(line) # A2E_MABU will deal with both stochastic and non-stochastic dust components
    s = line.split()
    if (len(s)>1):
        if (s[0].find('libmap')>=0): libmaps = s[1]
fpo.close()    


# if one is making a NN mapping (nnmake) and ini contained keyword nnthin, 
# feed A2E_MABU.py only absorbed[0::nnthin, :], data for every nnthin:th cell
# emitted.data contains all frequencies and maps can be written.... unless nnthin>1 !
if ((len(nnmake)>0)&(nnthin>1)):
    print("ASOC_driver, nnmake %s, nnthin %d .... fabs %s" % (nnmake, nnthin, fabs))
    H = np.fromfile(fabs, np.int32, 2)
    cells, nfreq = H
    # read and write in one go
    tmp  =  np.fromfile(fabs, np.float32)[2:].reshape(cells, nfreq)[0::nnthin, :]
    H[0] =  tmp.shape[0]  # new number of cells
    fabs = 'thin.abs'     # replaces original absorption file
    fp3  =  open(fabs, 'wb')
    H.tofile(fp3)
    np.asarray(tmp, np.float32).tofile(fp3)
    fp3.close()
    del tmp
    t0 = time.time()
    print("================================================================================")
    print('A2E_MABU.py %s %s %s %s' % (INI, fabs, femit, args))
    print("================================================================================")
    os.system('cp %s a2e_mabu_backup.ini' % INI)
    os.system('A2E_MABU.py %s %s %s %s' % (INI, fabs, femit, args))
else:
    t0 = time.time()
    print("================================================================================")
    print('A2E_MABU.py %s %s %s %s' % (INI, fabs, femit, args))
    print("================================================================================")
    os.system('cp %s a2e_mabu_backup.ini' % INI)
    os.system('A2E_MABU.py %s %s %s %s' % (INI, fabs, femit, args))


# Second SOC run -- compute maps 
# remove the "nomap" option and replace stochastic dusts with corresponding simple dusts

if ((len(nnmake)>0)&(nnthin>1)):
    print("nnmake + nnthin>1 => current emitted.data does not contain data for all cells")
    print(" => map making is skipped")
    print("   (or one should automatically use the created library to write emission and then make the maps)")
    sys.exit()
    
    
if (len(nnsolve)>0):
    # emission was solved with neural networds only for frequencies nnemit
    # make alternative ini file where dusts are limited to nnemit, dsc is replaces with nnemit.dsc
    print("*** INI FOR NNSOLVE MAPS ***")
    lines    = open(INI).readlines()
    fp3      = open('maps.ini', 'w')
    for line in lines:
        s = line.split()
        if (len(s)<1): continue
        if (s[0]=='nomap'): continue
        if (s[0]=='optical'):
            dust  = s[1]
            ndust = s[1]
            if (is_gs_dust(dust)): 
                # once again, use dust names without "gs_" in all SOC and A2E calculations !!
                # ... and ASOC will use only simple dust files
                ndust = '%s_simple.dust' % (dust.replace('gs_','').split('.')[0])
            ###
            # further modification for nnemit frequencies            
            nndust = write_nndust(ndust, nnemit, 'nnemit')  # only nnemit wavelengths
            line = line.replace(dust, nndust)
        elif (s[0]=='dsc'): # replace nnabs.csc with nnemit.csc
            line = 'dsc nnemit.dsc %s\n' % s[2]
        elif (s[0].find('backg')>=0):  # although this file is not read when making maps...
            nnbg  = np.fromfile(s[1], np.float32)[IND_nnemit]
            np.asarray(nnbg, np.float32).tofile(s[1]+'.nn')
            for k in range(len(s)):
                if (k==1):  fp3.write('%s.nn ' % s[k])
                else:       fp3.write('%s '    % s[k])
            fp3.write('\n')
            continue
        fp3.write(line)
        # ok, if INI contained libmaps line, the maps will be written for those frequencies only
    fp3.write('iterations 0\n')    
    fp3.write('nosolve    0\n') 
    fp3.close()            
        
else:
    # normal maps writing, with emitted containing all frequencies    
    lines    = open(INI).readlines()
    fp3      = open('maps.ini', 'w')
    for line in lines:
        s = line.split()
        if (len(s)<1): continue
        if (s[0]=='nomap'): continue
        if (s[0]=='optical'):
            dust = s[1]
            if (is_gs_dust(dust)): 
                # once again, use dust names without "gs_" in all SOC and A2E calculations !!
                # ... and ASOC will use only simple dust files
                line = line.replace(dust, '%s_simple.dust' % (dust.replace('gs_','').split('.')[0]))
        fp3.write(line)
        # ok, if INI contained libmaps line, the maps will be written for those frequencies only
    fp3.write('iterations 0\n')    
    fp3.write('nosolve    0\n') 
    fp3.close()            

print("================================================================================")
print("ASOC.py maps.ini")
print("================================================================================")
t0 = time.time()
os.system('ASOC.py maps.ini')
# print("... ASOC.py maps.ini... %.2f seconds" % (time.time()-t0))
