#!/usr/bin/julia

const BOLTZMANN = 1.38065e-16
const PLANCK    = 6.62607e-27
const C_LIGHT   = 2.99792e+10



function PrepareTdown(
  # Prepara cooling rates u -> u-1 into Tdown[NFREQ], a single grain size
  DUST::DustO,
  E::Array{Float64,1},               # number of energy bins
  FREQ:Array{Float64,1},             # FREQ[NFREQ]
  SKABS:Array{Float64,1}             # SKABS[NFREQ], corresponding pi*a^2*Q, current size only
  # return TDOWN:Array{Float64,2}  TDOWN[NE, NSIZE]
  )  
  Ef        =  PLANCK*FREQ
  NE        =  len(E)
  NEPO      =  NE+1
  NFREQ     =  length(FREQ)
  Tdown     =  zeros(Float64, NE)
  
  #   PrepareTdown(queue, [GLOBAL,], [LOCAL,], NFREQ, FREQ_buf, Ef_buf, SKABS_buf, NE, E_buf, T_buf, Tdown_buf)
  for u=2:NE    
    Eu   =  0.5*(E[u  ]+E[u+1])           # at most  u+1 = NE+1 = NEPO
    El   =  0.5*(E[u-1]+E[u  ])           # at least u-1 = 1
    Tu   =  Interpolate(NEPO, E, T, Eu)   # would be better if interpolated on log-log scale ?
    ee0  =  0.0 
    yy0  =  0.0 
    # Integral from 0.0 to Eu of    E^3*C/(exp(E/kT)-1)
    # First the full bins until end of the bin i Ef[i+1] exceeds upper limit Eu
    I    =  0.0 
    i    =  1    # current bin
    while ((Ef[i+1]<Eu) && (i<NFREQ))
      ee0  =  Ef[i] ;                                           # energy at the beginning of the interval (bin i)
      x    =  Interpolate(NFREQ, FREQ, SKABS, ee0/PLANCK)       # C at the beginning of the interval
      yy0  =  ee0*ee0*ee0* x /(exp(ee0/(BOLTZMANN*Tu))-1.0)     # integrand at the beginning of the interval
      for j=1:8                                                 # eight subdivisions of the bin i
        ee1  =  Ef[i] + j*(Ef[i+1]-Ef[i])/8.0                   # end of the sub-bin in energy
        x    =  Interpolate(NFREQ, FREQ, SKABS, ee1/PLANCK)     # C at the end of the interval
        yy1  =  ee1*ee1*ee1 * x / (exp(ee1/(BOLTZMANN*Tu))-1.0) # integrand at the end of the sub-bin
        I   +=  0.5*(ee1-ee0)*(yy1+yy0)                         # Euler integral of the sub-bin
        ee0  =  ee1                                             # next sub-bin starts at the end of previous 
        yy0  =  yy1
      end
      i += 1   # we have completed the integral till the beginning of the bin i ... now in ee0
    end
    # We have completed the integral up to the beginning of a bin that ends beyond Eu
    # The last partial step from Ef[i] to Eu
    if (Eu<Ef[NFREQ])
      for j=1:8
        ee1  =  Ef[i] + j*(Eu-Ef[i])/8.0
        x    =  Interpolate(NFREQ, FREQ, SKABS, ee1/PLANCK)
        yy1  =  ee1*ee1*ee1 * x / (exp(ee1/(BOLTZMANN*Tu))-1.0)
        I   +=  0.5*(ee1-ee0)*(yy1+yy0)
        ee0  =  ee1
        yy0  =  yy1
      end
    end
    # I *= 8.0*PI/((Eu-El)*C_LIGHT*C_LIGHT*PLANCK*PLANCK*PLANCK) ;
    # Warning:   8.0*PI/(C_LIGHT*C_LIGHT*PLANCK*PLANCK*PLANCK)  = 9.612370e+58
    I   *=  9.612370e+58 / (Eu-El) ;
    Tdown[u] = I ;
  end
  return Tdown
end





function Heating(
  DUST::DustO,
  E::Array{Float64, 1},         # E[NEPO], energy grid for the current size
  FREQ::Array{Float64,1},       # Ef[NFREQ], FREQ*PLANCK
  SKABS::Array{Float64,2}       # SKABS[NFREQ, NSIZE]
  )  
  # Precalculate integration weights for upwards transitions. Eq. 15, (16) in Draine & Li (2001)
  # With the weight precalculated, the (trapezoidal numerical) integral of eq. 15 is a dot product of the vectors
  # containing the weights and the absorbed photons (=Cabs*u), which can be calculated with (fused) multiply-add.
  # TODO: fix the treatment of the highest bin, although it probably won't make any difference...
  NEPO  =  length(E)
  NE    =  NEPO-1
  NFREQ =  length(FREQ)
  Ef    =  PLANCK*FREQ
  W     =  zeros(4)
  Z     =  zeros(Float64, NFREQ)   # vector of integration weights, each l->u
  L1    =  zeros(Int32, (NFREQ, NFREQ))
  L2    =  zeros(Int32, (NFREQ, NFREQ))
  Z     =  zeros(NFREQ, NE, NE)    # [NFREQ, u, l]
  no    =  0                       # number of integration weights
  for l=1:(NE-1)      
    El  =  0.5*(E[l]+E[l+1])
    dEl =  E[l+1]-E[l]
    for u=(l+1):NE            
      Eu    =  0.5*(E[u]+E[u+1]) 
      dEu   =  E[u+1]-E[u]             
      W[1]  =  E[u]-E[l+1] 
      W[2]  =  min( E[u]-E[l],  E[u+1]-E[l+1] ) 
      W[3]  =  max( E[u]-E[l],  E[u+1]-E[l+1] ) 
      W[4]  =  E[u+1] - E[l]       
      # calculate integral
      #   c*dEu/(Eu-El) *  Integral_W1^W4   Gul(E) * C(E) * u(E) * dE
      #       dEu is in the denominator of Gul => can be removed
      #   c/(Eu-El) *  Integral_W1^W4   Gul(E) * C(E) * u(E) * dE
      #       where Gul does not have dEu in the denominator
      # Integral [W1, W2]
      i = 1
      #=
      Euler integration over a fraction [a,b] of the full interval between i and (i+1).
      The integration weights for points i and i+1 are:
      ...  z[  i] = 0.5*(b-a)*(2-alpha-beta)*G1*C1
      ...  z[i+1] = 0.5*(b-a)*(alpha+beta)*G2*C2
      Here (G1,C1) and (G2,C2) are data at points i and i+1.
      alpha = (a-E[i])/(E[i+1]-E[i])
      beta  = (b-E[i])/(E[i+1]-E[i])
      =#
      # find bin index i such that Ef[i] < W[1] < Ef[i+1]
      while (Ef[i]<W[1])
        i += 1
      end
      i = max(i-1,1)      
      # integral over partial bin  [W1, Ef[i+1]]
      #  .... in theory could be [W[1], W[2]) inside bin i
      #  .... or we could have W[1]<Ef[1] or even W[2]<Ef[1]
      a, b   =  max(Ef[1], W[1]),   min(Ef[i+1], W[2])
      G1     =  (a-W[1])/dEl * Interpolate(NFREQ, FREQ, SKABS, a/PLANCK) 
      G2     =  (b-W[1])/dEl * Interpolate(NFREQ, FREQ, SKABS, b/PLANCK)
      alpha  =  (a-Ef[i])/(Ef[i+1]-Ef[i])
      beta   =  (b-Ef[i])/(Ef[i+1]-Ef[i])
      Z[i,   u, l]  +=  0.5*(b-a)*(2.0-alpha-beta)*G1
      Z[i+1, u, l]  +=  0.5*(b-a)*(alpha+beta)*G2      
      if (b==E[i+1])  # if step was till the end of a bin, we continue with the next bin
        i += 1
      end
      # at this point integral has been done up to  E[i] or already to W[2]
      # we continue in bin i, [b, W[2]] *or* [b, E[i+1]]
      while ((b<W[2])&(i<NFREQ))
        a       =  b
        G1      =  G2
        b       =  min(W[2], Ef[i+1])
        G2      =  (b-W[1])/dEl * Interpolate(NFREQ, FREQ, SKABS, b/PLANCK)
        alpha   =  (a-Ef[i])/(Ef[i+1]-Ef[i])
        beta    =  (b-Ef[i])/(Ef[i+1]-Ef[i])
        Z[i,  u, l]  +=  0.5*(b-a)*(2.0-alpha-beta)*G1
        Z[i+1, u, l] +=  0.5*(b-a)*(alpha+beta)*G2
        if (b<W[2])
          i += 1
        end
      end      
      # now the integral is ready up to W[2]
      # we start in bin i 
      while ((b<W[3])&(i<NFREQ)
        a       =  b
        G1      =  G2
        b       =  min(W[3], Ef[i+1])
        G2      =  min(dEl, dEu)/dEl * Interpolate(NFREQ, FREQ, SKABS, b/PLANCK)
        alpha   =  (a-Ef[i])/(Ef[i+1]-Ef[i])
        beta    =  (b-Ef[i])/(Ef[i+1]-Ef[i])
        Z[i,   u, l]   +=  0.5*(b-a)*(2.0-alpha-beta)*G1
        Z[i+1, u, l] +=  0.5*(b-a)*(alpha+beta)*G2
        if (b<W[3])
          i += 1
        end
      end      
      # integral ready up to W[3], we are in bin i
      while ((b<W[4])&(i<NFREQ))
        a       =  b
        G1      =  G2
        b       =  min(W[4], Ef[i+1])
        G2      =  (W[4]-b)/dEl * Interpolate(NFREQ, FREQ, SKABS, b/PLANCK)
        alpha   =  (a-Ef[i])/(Ef[i+1]-Ef[i])
        beta    =  (b-Ef[i])/(Ef[i+1]-Ef[i])
        Z[i,   u, l]  +=  0.5*(b-a)*(2.0-alpha-beta)*G1
        Z[i+1, u, l]  +=  0.5*(b-a)*(alpha+beta)*G2
        if (b<W[4])
          i += 1
        end
      end
      # check the first and last nonzero elements => L1[u,l], L2[u,l]
      i = 1
      while ((Z[i, u, l]==0.0)&(i<NFREQ))
        i += 1
      end
      L1[u,l] = i
      i = NFREQ
      while ((Z[i, u, l]==0.0)&(i>1))
        i -= 1
      end
      L2[u,l] = i
      no  +=  L2[u,l] -L1[u,l] + 1
      
    end # for  u
  end # for l
  return no, L1, L2, Z
  
end # PrepareIntegrationWeights




# A2E_pre.jl  gset-dist  frequencyfile solverfile   [NE]
DUST   =  DustO()
Read(DUST, args[2])
FREQ   =  readdlm(args[3])[:,0]
NFREQ  =  length(FREQ)
NSIZE  =  DUST.NSIZE
NE     =  128
if (length(args)>4)
  NE   =  parse(Int32, args[4])
end
NEPO   =  NE+1



# Write the solver file
#  in Python SKABS[NSIZE, NFREQ] ..... in Julia SKABS[NFREQ, NSIZE]

fp = open("$label.heat", "wb")  
fp.write(Int32(NFREQ))
fp.write(Array{Float32}(FREQ))
fp.write(Float32(GRAIN_DENSITY))
fp.write(Int32(NSIZE))
fp.write(Array{Float32}(SFRAC/GRAIN_DENSITY))  #  sum(SFRAC/GRAIN_DENSITY)==1, no GRAIN_DENSITY !!
fp.write(Int32(NE))
fp.write(Array{Float32}(SKABS))                #  SKAbs_Int()  ~   pi*a^2*Qabs * SFRAC, including GRAIN_DENSITY


for isize=1:NSIZE
  SKABS         =  SKabs_Int(DUST, isize, FREQ)
  # Create temperature & energy grid based on temperature limits in DustO
  tmin, tmax    =  DUST.TMIN[isize], DUST.TMAX[isize]
  T             =  tmin+(tmax-tmin)*(range(0.0, stop=NEPO-1.0, length=NEPO)/(NEPO-1.0))^2.0
  E             =  E2T(DUST, isize, T)      
  ##
  n, L1, L2, Z  =  Heating(DUST, E, FREQ, SKABS)
  write(fp, Int32(n))                 # number of non-zero integration weights
  for l=1:(NE-1)
    for u=(l+1):NE
      Z[ L1[u,l] : L2[u,l] , u, l]    # integration weights themselves
    end
  end
  write(fp, Array{Float32,1}(L1))     # L1[u,l] == u runs faster in the file
  write(fp, Array{Float32,1}(L2))
  # Finally Tdown for the current size
  Tdown  =  PrepareTdown(DUST, E, FREQ, SKABS)   # Tdown[NE]
  fp.write(Array{Float32}(Tdown))
end
close(fp)








#!/usr/bin/julia

using Printf
using DelimitedFiles
using OpenCL

# using PyPlot

#=
(1) Read DustEM files
(2) write data for SOC run
(3) write data for pyA2E.py run

Input is file like dustem_CM20.dust which refers to original DustEM
files and is used to define all dust properties.
=#


DUSTEM_DIR = "/home/mika/tt/dustem4.0_web/"


mutable struct DData
  NAME
  ##########################
  LOGN::Bool
  PLAW::Bool
  ED::Bool
  CV::Bool
  a0::Float64
  sigma::Float64
  alpha::Float64
  a_t::Float64
  a_c::Float64
  gamma::Float64
  a_u::Float64
  z::Float64
  eta::Float64
  GRAIN_DENSITY::Float64       # total number of grains per H
  ##########################
  NE::Int32
  NEPO::Int32
  TMIN::Array{Float64,1}
  TMAX::Array{Float64,1}
  ## TDOWN::Array{Float32,2}      #  [NE, NSIZE]   (ie runs faster!)
  CE::Array{Float64,2}         #  [CNT, CNSIZE]
  E::Array{Float64, 2}         #  [NEPO, NSIZE]
  ##########################
  NSIZE::Int32
  RMASS::Float64
  RHO::Float64
  AMIN::Float64
  AMAX::Float64
  SIZE_A::Array{Float64,1}       # discrete sizes
  ## SIZE_F::Array{Float64,1}       # dn/da at SIZE_A sizes
  SCALE::Float64
  S_FRAC::Array{Float64,1}    # (dn/da)*A at SIZE_A -- after normalisation number of grains per size bin
  ###  S_FRAC::Array{Float64,1}       # fraction of grains (number) in size bin == CRT_SFRAC
  FREQ::Array{Float64,1}         # the grid used in RT
  Ef::Array{Float64,1}           # == PLANCK*FREQ
  ## Optical data
  QFREQ::Array{Float64,1}   # [freq]
  QNFREQ::Int32
  QNSIZE::Int32
  QSIZE::Array{Float64,1}   # [size]
  #QABS::Array{Float64,2}    # [freq,size]
  #QSCA::Array{Float64,2}    # [freq,size]
  #G::Array{Float64,2}       # [freq,size]
  OPT::Array{Float64,3}     # [size, freq, 4]
  BINS::Int32
  ## Heat capacities
  CNSIZE::Int32
  CNT::Int32
  CSIZE::Array{Float64,1}   # [size]
  CT::Array{Float64,1}      # [CNT]
  CC::Array{Float64,2}      # [CNT, CNSIZE]
  DData() = new()
end

const FACTOR           =  1.0e20           # MUST BE THE SAME AS IN A2E !!
const C_LIGHT          =  2.99792458E10
const AMU              =  1.6605E-24
const H_K              =  4.7995074E-11
const BOLZMANN         =  1.3806488e-16
const BOLTZMANN        =  1.3806488e-16
const BOLTZMANN_SI     =  1.3806488e-23
const STEFAN_BOLTZMANN =  5.670373e-5
const SB_SI            =  5.670373e-8
const CGS_TO_JY_SR     =  1e23          # erg/cm2/sr/Hz = CGS_TO_JY_SR * Jy/sr
const PLANCK           =  6.62606957e-27
const PLANCK_SI        =  6.62606957e-34
const M0               =  1.99e33
const MJupiter         =  1.9e30        # [g]
const GRAV             =  6.67e-8
const GRAV_SI          =  6.673e-11
const PARSEC           =  3.0857E18
const ELECTRONVOLT     =  1.6022e-12
const AU               =  149.597871e11
const RSUN             =  6.955e10
const RSUN_SI          =  6.955e8
const DSUN             =  1.496e13  # cm
const DSUN_SI          =  1.496e11  # 1.496e8 km
const MSUN             =  1.9891e33
const MSUN_SI          =  1.9891e30
const M_EARTH          =  5.972e27
const LSUN             =  3.839e33
const LSUN_SI          =  3.839e26
const TSUN             =  5778.0
const MJUPITER         =  1.9e30
const H_C2             =  PLANCK/(C_LIGHT*C_LIGHT)
const H_C2_GHz         =  PLANCK/(C_LIGHT*C_LIGHT)*1.0e27
const ARCSEC_TO_DEGREE =  (1.0/3600.0)
const DEGREE_TO_RADIAN =  0.0174532925199432958
const ARCMIN_TO_RADIAN =  (2.9088820e-4)
const ARCSEC_TO_RADIAN =  (4.8481368e-6)
const HOUR_TO_RADIAN   =  (0.261799387)
const MINUTE_TO_RADIAN =  (4.3633231e-3)
const SECOND_TO_RADIAN =  (7.2722052e-5)
const RADIAN_TO_DEGREE =  57.2957795130823208768
const RADIAN_TO_ARCMIN =  3437.746771
const RADIAN_TO_ARCSEC =  206264.8063
const RADIAN_TO_HOUR   =  3.819718634
const RADIAN_TO_MINUTE =  229.1831181
const RADIAN_TO_SECOND =  13750.98708
const ARCMIN_TO_DEGREE =   (1.0/60.0)
const DEGREE_TO_ARCMIN =   60.0
const DEGREE_TO_ARCSEC =   3600.0
const BSD              =   0.94e21    #  N(H2)=0.94e21*Av
const mH               =   1.0079*AMU

function logspace(a, b, n)
  return 10.0 .^ Array(range(log10(a), stop=log10(b), length=n))
end

function linspace(a, b, n)
  return Array(range(a, stop=b, length=n))
end

function um2f(um)
  return (C_LIGHT/1.0e-4)./um
end  

function f2um(f)
  return (1.0e4*C_LIGHT)./f
end

function tof(x)
  return parse(Float64, x)
end

function toi(x)
  return parse(Int32, x)
end

function PlanckIntensity(freq::Real, T::Real)
  return (2.0*PLANCK*(freq/C_LIGHT)^2.0*freq) / (exp(H_K*freq/T)-1.0)
end


function LIP(x::Array{<:Real,1}, y::Array{<:Real,1}, x0::Real)::Float64
  # Linear interpolation from discrete values.
  if (x[2]>x[1])
    # Values are in increasing order
    if (x0<x[1])
      return 0.0
    elseif (x0>x[end])
      return 0.0
    else
      for i in 2:length(x)
        if (x[i]>=x0)
          w  =  (x[i]-x0)/(x[i]-x[i-1])
          return w*y[i-1] + (1.0-w)*y[i]
        end
      end
    end
    return 0.0
  else
    # x is in decreasing order
    if (x0<x[end])
      return 0.0
    elseif (x0>x[1])
      return 0.0
    else
      for i in 2:length(x)
        if (x[i]<=x0)
          w  =  (x0-x[i])/(x[i-1]-x[i])
          return w*y[i-1] + (1.0-w)*y[i]
        end
      end
    end
    return 0.0    
  end
end


function LIP(x::Array{<:Real,1}, y::Array{<:Real,1}, X0::Array{<:Real,1})::Array{Float64,1}
  # Linear interpolation from discrete values.
  res = zeros(Float64, length(X0))
  for i in 1:length(X0)
    res[i] = LIP(x, y, X0[i])
  end
  return res
end


function LIN(x::Array{<:Real,1}, y::Array{<:Real,1}, a::Real, b::Real)
  # Integral of linear interpolated function, no extrapolation outside x[] coverage.
  if false
    if ((a>x[end])|(b<x[1]))
      return 0.0
    end
  else
    a = clamp(a, x[1], x[end])
    b = clamp(b, x[1], x[end])
  end
  I  = 0.0
  w  = 0.0
  i  = 2
  while ((x[i]<a) & (i<(length(x))))
    i += 1
  end
  # a is between x[i-1] and x[i]; figure out first piece till the next bin boundary
  aa   =   max(x[i-1], a)
  w    =   (x[i]-aa)/(x[i]-x[i-1])  # a<x[1] .... start integral only from x[1] !!
  y1   =   w*y[i-1] + (1.0-w)*y[i]
  if (x[i]>=b)  # whole integral within the same bin
    w  =  (x[i]-b)/(x[i]-x[i-1])
    y2 =  w*y[i-1] + (1.0-w)*y[i]
    I +=  (b-a)*0.5*(y1+y2)
    return I
  else        # first step extends to x[i] and beyond
    I +=  (x[i]-aa)*0.5*(y1+y[i])
  end
  # we have dealt with the integral up to x[i]
  i += 1
  # rest of the integral from x[i] to b
  while (x[i]<b)                         # full steps x[i-1] to x[i]
    I += 0.5*(x[i]-x[i-1])*(y[i]+y[i-1])
    i += 1
    if (i>length(x))   #  b > x[end]
      return I
    end
  end
  #  x[i]>b --- last partial step [x[i-1],b]
  w   =  (x[i]-b)/(x[i]-x[i-1])
  y1  =  w*y[i-1] + (1.0-w)*y[i]
  I  +=  0.5*(b-x[i-1])*(y[i-1]+y1)
  # @printf("B:  %6.3f < %6.3f < %6.3f\n", x[i-1], b, x[i])
  return I
end  



if false
  # @@ Test integration
  x  = linspace(0.0, 2.0, 50)
  y  = sin.(x)  
  ## y  = 1.0*x
  N  = 10000
  R1 = zeros(N)
  R2 = zeros(N)
  for i in 1:N
    aa  = -0.2 + 2.4*rand()
    bb  = -0.2 + 2.4*rand()
    a, b = min(aa,bb), max(aa,bb)
    r1 =  LIN(x, y, a, b)
    r2 =  -(cos(clamp(b,0.0,2.0))-cos(clamp(a,0.0,2.0)))
    ## r2 =  0.5*(  clamp(b,0.0,2.0)^2.0 - clamp(a,0.0,2.0)^2.0 )
    R1[i] = r1
    R2[i] = r2
    if ((abs(r1-r2)>0.001))
      @printf("LIN = %12.4e  ~  %12.4e   [%6.3f,%6.3f]\n", r1, r2, a, b)
    else
      @printf("LIN = %12.4e  ~  %12.4e\n", r1, r2)
    end
  end
  plot(R1, R2, marker=".", linestyle="")
  show()
  exit(0)
end





function size_distribution(a::Real, dd::DData)
  #  Based on dust parameters in dd (DData structure), return the
  #  "SIZE_F" = dn/da, number of grains per linear size interval.
  println("size_distribution(Real,DData)")
  res = 0.0
  if ((a<dd.AMIN)|(a>dd.AMAX))
    return 0.0
  end
  if     (dd.LOGN)
    res = exp(- 0.5*(log(a/dd.a0)/dd.sigma)^2.0 ) / a  #  1/a to convert dn/dloga to dn/da MUST HAVE LOG GRID!!
  elseif (dd.PLAW)
    res = a^dd.alpha
  end
  if     (dd.ED)
    if (a>dd.a_t)
      res *= exp(-((a-dd.a_t)/dd.a_c)^dd.gamma)
    end
  end
  if     (dd.CV)
    res *= (1.0 + abs(dd.z)*(a/dd.a_u)^dd.eta)^(sign(z))
  end
  return dd.SCALE*res   # these are dn/da for the discrete size values in a
end



function size_distribution(a::Array{<:Real,1}, dd::DData)
  #  Based on dust parameters in dd (DData structure), return the
  #  "SIZE_F" = dn/da, number of grains per linear size interval.
  println("size_distribution(Array,DData)\n")
  res  =  zeros(Float64, length(a))
  if     (dd.LOGN)
    res       = exp(- 0.5*(log(a./dd.a0)./dd.sigma).^2.0 ) ./ a
  elseif (dd.PLAW)
    res       = a.^dd.alpha
  end
  if     (dd.ED)
    ind       = findall(x->x>=dd.a_t, a)
    res[ind] .*= exp.(-((a[ind].-dd.a_t)./dd.a_c).^dd.gamma)
  end
  if     (dd.CV)
    res     .*= (1.0 + abs(dd.z)*(a/dd.a_u).^dd.eta).^(sign(z))
  end
  ind       = findall(x->((x<(0.9999*dd.AMIN))|(x>(1.0001*dd.AMAX))), a)
  res[ind] .= 0.0
  ##
  return dd.SCALE.*res
end # size_distribution




#####################################################################################################


  
function read_DM_sizes(DD, sizefile; tmin=4.0, tmax1=40.0, tmax2=140.0)
  """
  Read size information from a DutsEM text file.
  **ASSUME SIZES GIVEN BY ANALYTICAL FORMULAS**
  The appropriate line starts with the name of this dust component.
  """
  println("read_DM_sizes($sizefile) -- ", DD.NAME)
  NSIZE = 1024  # intermediate discretisation of the size distribution
  fp    = open(sizefile, "r")
  ln    = ""
  TYPE  = ""
  s     = []
  for ln in eachline(fp)
    s   = split(ln)
    if (s[1]!=DD.NAME)  # this line not for this dust
      continue 
    end
    # found the correct line
    DD.NSIZE = toi(s[2])
    TYPE     = s[3]
    DD.RMASS = tof(s[4])
    DD.RHO   = tof(s[5])
    DD.AMIN  = tof(s[6])
    DD.AMAX  = tof(s[7])
    break
  end
  close(fp)
  if (TYPE=="size")
    println("Dust sizes must be give with analytical formulas!")
    exit(0)
  end
  # We should be using analytical formulas for the size.
  #  ... discretise here but on a large anough arrays
  i = 8  # next parameter on the line
  DD.LOGN = false
  DD.PLAW = false
  DD.ED   = false
  DD.CV   = false
  stype   = split(TYPE, '-')
  for ss in stype  # the actual size distribution type is somewhere in the string, separated by '-'
    if (ss=="logn")     # TYPE=1
      # dn/dloga ~  exp(-( log(a/a0)/sigma )**2)
      DD.LOGN  =  true
      DD.a0    =  tof(s[i])   # a0
      DD.sigma =  tof(s[i+1]) # sigma
      i        =  i+2
      #@printf("*** logn ***  a0 %10.3e, sigma %10.3e", DD.a0, DD.sigma)
    elseif (ss=="plaw") # TYPE 2
      # dn/da ~ a^alpha
      DD.PLAW  =  true
      DD.alpha =  tof(s[i])  # alpha
      i        =  i+1
      #@printf("*** plaw *** alpha %10.3e\n", DD.alpha)
    elseif (ss=="ed")   # TYPE = 3
      #  *= exp(-( (a-at)/ac )**gamma), for a>at
      DD.ED    =  true
      DD.a_t   =  tof(s[i])      # a_t
      DD.a_c   =  tof(s[i+1])    # a_c
      DD.gamma =  tof(s[i+2])    # gamma
      i        =  i+3
      #@printf("*** ed   *** at %10.3e ac %10.3e gamma %10.3e\n", DD.a_t, DD.a_c, DD.gamma)
    elseif (ss=="cv")   # TYPE = 4
      DD.CV    =  true
      #  *=  (1+|z|*(a/au)**eta)**sign(z)
      @printf("*** cv   ***")
      DD.a_u   =  tof(s[i])    # a_u
      DD.z     =  tof(s[i+1])  # z
      DD.eta   =  tof(s[i+2])  # eta
      i        =  i+3
      #@printf("*** cv *** a_u %10.3e,  z %10.3e, eta %10.3e\n", DD.a_u, DD.z, DD.eta)
    end
  end # if -- basic size distribution
  
  # Discretised values
  DD.SCALE       = 1.0
  DD.SIZE_A      = logspace(DD.AMIN, DD.AMAX, DD.NSIZE)
  SIZE_F         = size_distribution(DD.SIZE_A, DD)   # SIZE_F = dn/da at SIZE_A sizes
  # Normalisation wrt H
  #     volume_integral * rho = Hydrogen_mass * RMASS
  mass_integrand =  Array{Float64}(  ((4.0*pi/3.0)*DD.RHO) * (SIZE_F .* (DD.SIZE_A.^3.0))  )
  # res = integral of mass_integrand from AMIN to AMAX
  # res, dres      =  quad(ip, self.AMIN, self.AMAX, epsrel=1.0e-10, epsabs=EPSABS2)  # total dust mass..$
  res            =  LIN(DD.SIZE_A, mass_integrand, DD.AMIN, DD.AMAX) # note SIZE_F -> integrad ()/cm in size
  SIZE_F        *=  mH*DD.RMASS / res
  # Add this information to DD so that it will be included in the values returned by size_distribution()
  DD.SCALE       =  mH*DD.RMASS / res
  # Do CRT-type summation for the integral, as in the python version (and CRT)
  DD.S_FRAC      =  SIZE_F .* DD.SIZE_A  # ASSUMES LOGARITHMIC BINS !!!
  vol            =  (4.0*pi/3.0) * sum( DD.S_FRAC .* DD.SIZE_A.^3.0 )
  DD.S_FRAC     *=  mH*DD.RMASS/(DD.RHO*vol)
  # Calculate the total number of grains per H ... should be given by size_distribution once SCALE was set
  #  ... is also  included in SIZE_F .... and is included in CRT_SFRAC
  # DD.GRAIN_DENSITY = LIN(DD.SIZE_A, DD.SIZE_F, DD.AMIN, DD.AMAX)
  DD.GRAIN_DENSITY = LIN(DD.SIZE_A, SIZE_F, DD.AMIN, DD.AMAX)
  println("GRAIN_DENSITY ", DD.GRAIN_DENSITY)
  # double check
  if false
    vol = sum(DD.S_FRAC .* DD.SIZE_A.^3.0) * 4.0*pi/3.0
    @printf("Dust mass %10.3e, H mass %10.3e, ratio %10.3e, RMASS %10.3e\n", vol*DD.RHO, mH, vol*DD.RHO/mH, DD.RMASS)
  end
  # Calculate the number of grains in each size bin  = S_FRAC
  DD.S_FRAC      = zeros(Float32, DD.NSIZE)
  for isize in 1:DD.NSIZE
    DD.S_FRAC[isize] = SIZE_F[isize] * DD.SIZE_A[isize]   # ~ integral, taking into account log spacing
  end
  DD.S_FRAC     /= sum(DD.S_FRAC)          # fraction of grains in each size bin 
  ###
  fp = open("sizes.dump", "w")
  write(fp, Array{Float32}(SIZE_F))
  write(fp, Array{Float32}(DD.S_FRAC))
  close(fp)
  
  # Check if we also had mixing -- to be applied after the above normalisation
  if ("mix" in stype)
    println("***MIX***")
    mixfile = @sprintf("%s/data/MIX_%s.DAT",  DUSTEM_DIR, DD.NAME)
    mix     = readdlm(mixfile)[:]
    if (length(mix)==DD.NSIZE)  # using original dustem grid !!
      println("mix has the same NSIZE")
      SIZE_F     *= mix
      DD.S_FRAC  *= mix
    else
      # mix file has different length... but should still correspond to logarithmic sampling
      # from AMIN to AMAX
      println("mix has different NSIZE --- interpolate")
      xx  = logspace(DD.AMIN, DD.AMAX, length(mix))
      # interpolate values (xx, mix) to (DD.SIZE_A, new_mix)
      yy             = LIP(xx, mix, DD.SIZE_A)
      SIZE_F    .*= yy
      DD.S_FRAC .*= yy
    end
  end # mix
  
  
  # ???
  D.NE    =  128
  D.NEPO  =  D.NE+1
  D.TMIN  =  ones(Float64, D.NSIZE)*tmin
  D.TMAX  =  zeros(Float64, D.NSIZE)
  for isize = 1:(D.NSIZE)   # isize increases, TMAX decreses
    D.TMAX[isize]  =  tmax1 + (((D.NSIZE-0.9999-isize)/(D.NSIZE-0.9999))^2) * (tmax2-tmax1)
    @printf(" size %3d --- TMIN %7.2f   TMAX %7.2f\n", isize, D.TMIN, D.TMAX[isize])
  end
  

end  # --- read_DM_size



function read_DM_frequency(DD, lambdafile)
  # Start by reading the frequency grid
  # lambdafile = @sprintf("%s/oprop/LAMBDA.DAT", DUSTEM_DIR)
  println("Reading frequencies from $lambdafile")
  um         = readdlm(lambdafile, skipstart=4)[:]
  DD.QFREQ   = um2f(um)
  DD.QNFREQ  = length(DD.QFREQ)
end  


  
function read_DM_optical(DD::DData, optfile, gfile)
  """
  Read optical data from DustEM file in DUSTEM_DIR/oprop
  """
  # Read the Q file
  # optfile = @sprintf("%s/oprop/Q_%s.DAT", DUSTEM_DIR, DD.NAME)
  println("Reading optical data from $optfile")
  fp      = open(optfile)
  lines   = readlines(fp)
  close(fp)
  ###
  ln = ""
  iline = 1
  while((length(lines[iline])<2) | (lines[iline][1]=='#'))
    iline += 1
  end
  DD.QNSIZE = toi(split(lines[iline])[1])
  DD.QSIZE  = zeros(Float64, DD.QNSIZE)
  s         = split(lines[iline+1])
  if (length(s)!=DD.QNSIZE)
    @printf("[1] Error reading optical data -- line has %d != QNSIZE=%d\n ", length(s), DD.QNSIZE)
    exit(0)
  end
  for i in 1:(DD.QNSIZE)
    DD.QSIZE[i] = tof(s[i])
  end
  DD.QSIZE *= 1.0e-4  # conversion um -> cm
  # Read Qabs and Qsca data (increasing order of wavelength!)
  x      =  readdlm(optfile, skipstart=iline+3, comments=true, comment_char='#')
  QABS   =  x[1:DD.QNFREQ,     :]     #  DD.QABS[freq, size]
  QSCA   =  x[(1+DD.QNFREQ):end, :]     
  ok     =  (DD.QNFREQ, DD.QNSIZE)
  if ( (size(QABS)!=ok) | (size(QSCA) != ok) )
    println("[2] Error reading Qabs, Qsca", size(QABS), size(QSCA), ok)
    exit(0)
  end
  # Read the g parameters -- assume the grid is the same as for Q
  # gfile   =  @sprintf("%s/oprop/G_%s.DAT", DUSTEM_DIR, DD.NAME)
  G    =  readdlm(gfile, skipstart=9)
  if (size(G)!=ok)
    println("[3] Error reading g ", size(G), ok)
  end
  # Put data to DD.OPT[size, freq]
  DD.OPT = zeros(DD.QNSIZE, DD.QNFREQ, 4)
  for isize in 1:DD.QNSIZE
    for ium in 1:DD.QNFREQ
      DD.OPT[isize, ium, :] = [ f2um(DD.QFREQ[ium]), QABS[ium, isize], QSCA[ium, isize], G[ium, isize] ]
    end
  end
end




function read_DM_heat(D, cfile)
  """
  Read heat capacities from DustEM file in DUSTEM_DIR/hcap
  """
  lines = readlines(open(cfile))
  i     = 1
  while(lines[i][1]=='#')
    i += 1
  end
  # this line has the number of sizes
  D.CNSIZE = toi(split(lines[i])[1])
  i += 1
  D.CSIZE = zeros(Float64, D.CNSIZE)
  s = split(lines[i])
  for isize in 1:length(D.CNSIZE)
    D.CSIZE[isize] = tof(s[isize]) * 1.0e-4  # file [um], CSIZE [cm]
  end
  i += 1
  # this line has the number of T values
  D.CNT = toi(split(lines[i])[1])
  # the rest, first column = log(T), other columns = C [erg/K/cm3] for each size
  #   self.ClgC[iT, iSize]
  d = readdlm(cfile, skipstart=i)
  ClgT = d[:,1]       #  [CNT]             --- yes, they are log_10
  ClgC = d[:,2:end]   #  [CNT, CNSIZE]
  # @printf("CNSIZE %d, CNT %d, skipstart %d\n", D.CNSIZE, D.CNT, i)
  if (size(ClgC) != (D.CNT, D.CNSIZE))
    println("Error reading enthalpy file !!")
    exit(0)
  end
  D.CT = 10.0.^ClgT
  D.CC = 10.0.^ClgC     #    erg/cm3/K       [CNT, CNSIZE]
  # Precalculate  E for a grid of temperatures [TMIN, TMAX],
  D.CE = zeros(Float64, D.CNT, D.CNSIZE)
  for iT in 1:D.CNT
    # integral of (CT)*dT  from 0 to CT[iT] over temperature
    for isize in 1:D.CNSIZE
      D.CE[iT, isize] =  LIN(D.CT, D.CC[:, isize], D.CT[1], D.CT[iT])   # erg/cm3
      # we DO NOT YET MULTIPLY BY VOLUME to avoid the need for that interpolation
    end
  end
  # Now we should be able to do transformation E->T and T->E by interpolating from these tables
  #   -->  E2T() and T2E()

  # Set up energy bins ---  E[NEPO, NSIZE], where size is D.NSIZE and D.SIZE_A
  D.E = zeros(Float64, D.NEPO, D.NSIZE)  # --- values at bin borders
  for isize in 1:D.NSIZE
    # size increases, tpeak decreases...
    emin   =  T2E(D, D.SIZE_A[isize], D.TMIN[isize])
    emax   =  T2E(D, D.SIZE_A[isize], D.TMAX[isize])
    for ie  in 1:D.NEPO      
      D.E[ie, isize] = exp(log(emin)+(ie/D.NE)*(log(emax)-log(emin)))
    end
  end
end

  




############################################################################################################




function read_TRUST_sizes(D, sizefile)
  """
  Read size distribution and set
  AMIN, AMAX =   limits of the grain sizes [cm]
  SIZE_A     =   discrete sizes [cm]
  SIZE_F     =   values dn/da at the sizes SIZE_A (with final normalisation)
  S_FRAC     =   number of grains per SIZE_A bin
  GRAIN_DENSITY = total number of grains = integral over SIZE_F from AMIN to AMAX
  """
  d = readdlm(sizefile, comments=true)
  # the first line is NSIZE, NE
  # the remaining lines a[um], f(a) [cm^-1 H^-1] --- already the final normalisation!!
  # D.NSIZE  =  Int32(d[1,1])
  # D.NE     =  Int32(d[1,2])   --- this defined in the CRT-type dust file that referred to TRUST files?
  # the rest are directly SIZE_A and SIZE_F
  D.SIZE_A =  copy(d[2:end, 1])  # skip first row = (nsizes, nebins)
  SIZE_F   =  copy(d[2:end, 2])
  D.NSIZE  =  length(D.SIZE_A)
  D.S_FRAC =  D.SIZE_A.*SIZE_F   # proportional to the number of grains per bin 
  D.AMIN   =  D.SIZE_A[1]*1.0001 
  D.AMAX   =  D.SIZE_A[end]*0.9999
  # GRAIN_DENSITY by integration
  #  at this point S_FRAC is still differential f(a)  cm-1 / H
  # Calculate total number of grains as integral
  D.GRAIN_DENSITY = LIN(D.SIZE_A, SIZE_F, D.AMIN, D.AMAX)
  # *IF* size intervals are logarithmic, S_FRAC*A is proportional to the number of grains per bin
  D.S_FRAC = D.S_FRAC .* D.SIZE_A
  # final normalisation to make sure sum(D.S_FRAC) == GRAIN_DENSITY
  tmp      = sum(D.S_FRAC)
  D.S_FRAC .*=  GRAIN_DENSITY/tmp
end



function read_TRUST_optical(D::DData, optfile)
  """
  Read optical data (Q, g) and the corresponding frequency grid QFREQ
  ...  QABS[QNFREQ, QNSIZE]
  ...  OPT[QNSIZE, QNFREQ, 4] = { QFREQ, QABS, QSCA, G }
  ...  like in the case of DustEM, frequencies are in DECREASING ORDER !
  Note:  TRUST did not specify scattering function ?????
  ...... use g parameters from the optical file.
  """
  fp = open(optfile, "r")
  readline(fp)
  s = split(readline(fp))     #  QNSIZE, QNFREQ
  D.QNSIZE = toi(s[1])
  D.QNFREQ = toi(s[2])
  D.QSIZE  = zeros(Float64, D.QNSIZE)
  D.QFREQ  = zeros(Float64, D.QNFREQ)
  # the rest of the file   skip + size + skip + {QNFREQ lines}
  D.OPT    =  zeros(Float64, D.QNSIZE, D.QNFREQ, 4)
  for isize in 1:D.QNSIZE
    readline(fp)
    D.QSIZE[isize] = tof(split(readline(fp))[1])
    readline(fp)
    for ium in 1:D.QNFREQ
      s                     =  split(readline(fp))
      D.QFREQ[ium]          =  um2f(tof(s[2]))   # keep QFREQ also as a separate array
      D.OPT[isize, ium, 1]  =  um2f(tof(s[2]))   # freq
      D.OPT[isize, ium, 2]  =  tof(s[3])         # Qabs
      D.OPT[isize, ium, 3]  =  tof(s[4])         # Qsca
      D.OPT[isize, ium, 4]  =  tof(s[6])         # g
    end
  end
  D.QSIZE *= 1.0e-4  # file was [um], we need [cm]
end


function read_TRUST_heat(D, cfile)
  """
  Read enthalpy information.
  TRUST file, fourth line:    TMIN, TMAX, NT, bulk density
  The remaining lines:        T,   E=erg/g,    C=erg/g/K
  Note: Dustem had data separately for different grain sizes (with different C??).
  We (unnecessarily) use a size grid taken from optical data gridding.
  """
  fp = open(cfile, "r")
  readline(fp)
  readline(fp)
  readline(fp)
  s = split(readline(fp))
  tmin, tmax, D.CNT, D.RHO = tof(s[1]), tof(s[2]), toi(s[3]), tof(s[4])
  close(fp)
  # the rest of the file = {  T [K], E [erg/g], C [erg/g/K]  }
  d        =  readdlm(cfile, skipstart=4)
  D.CT     =  copy(d[:,1])
  E        =  d[:,2]    # erg/g    ... used for the actual array D.CE
  C        =  d[:,3]    # erg/g/K  ... used for the actual array D.CC
  ###
  # NOTE !!   D.CC is probably not needed
  #           D.CE is needed only in T2E() and E2T() .... where CE == erg/cm3 !!
  #           ... since TRUST has C [erg/g/K] and E [erg/g] independent of the grain size,
  #               we could skip size discretisatio altogether
  D.CNT    =  length(D.CT)
  D.CNSIZE =  copy(D.QNSIZE)   # use the same size grid as optical data (which must be read first)
  D.CSIZE  =  copy(D.QSIZE)
  D.CE     =  zeros(Float64, D.CNT, D.CNSIZE)  # [ CNT, CNSIZE ] --- different from E[NEPO, NSIZE] !!
  D.CC     =  zeros(Float64, D.CNT, D.CNSIZE)  # [ CNT, CNSIZE ]
  for isize in 1:D.CNSIZE
    for it in 1:D.CNT
      D.CC[it, isize] = C[it] * D.RHO   #  erg/g/K * g/cm3  = erg/cm3/K
      D.CE[it, isize] = E[it] * D.RHO   #  erg/g   * g/cm3  = erg/cm3
    end
  end
end


############################################################################################################



function read_GSET_sizes(D, sizefile)
  """
  Read size distribution and set
  AMIN, AMAX =   limits of the grain sizes [cm]
  SIZE_A     =   discrete sizes [cm]
  SIZE_F     =   values dn/da at the sizes SIZE_A (with final normalisation)
  S_FRAC     =   (dn/da)*a, with the same normalisation as SIZE_F
  GRAIN_DENSITY = total number of grains = integral over SIZE_F from AMIN to AMAX
  """
  fp              =  open(sizefile)
  # first line = GRAIN_DENSITY
  D.GRAIN_DENSITY =  tof(split(readline(fp))[1])
  # second line = NSIZE, NE
  s               =  split(readline(fp))
  D.NSIZE         =  toi(s[1])
  D.NE            =  toi(s[2])
  D.NEPO          =  1+D.NE
  # rest of the file:   a [um],  S_FRAC,   Tmin,   Tmax
  d               =  readdlm(sizefile, skipstart=3)
  D.SIZE_A        =  copy(d[:,1]) * 1.0e-4    # [cm]
  D.S_FRAC        =  copy(d[:,2])             # directly fraction of grains in bin i !!!
  D.TMIN          =  copy(d[:,3])
  D.TMAX          =  copy(d[:,4])
  # WITHIN THESE SCRIPTS S_FRAC IS DIRECTLY THE NUMBER OF GRAINS PER SIZE INTERVAL
  D.S_FRAC      .*=  D.GRAIN_DENSITY          # => grains per bin per H
  ## D.SIZE_F     =  Array{Float64,1}([])
  # WE STILL NEED SIZE_F FOR THE DSF CALCULATIONS ??? OR NEED TO REWRITE THOSE
  D.AMIN          =  D.SIZE_A[1]*1.0001 
  D.AMAX          =  D.SIZE_A[end]*0.9999
end



function read_GSET_optical(D::DData, optfile)
  """
  Read optical data (Q, g) and the corresponding frequency grid QFREQ
  ...  QABS[QNFREQ, QNSIZE]
  ...  OPT[QNSIZE, QNFREQ, 4] = { QFREQ, QABS, QSCA, G }
  ...  like in the case of DustEM, frequencies are in DECREASING ORDER !
  Note:  TRUST did not specify scattering function ?????
  ...... use g parameters from the optical file.
  """
  fp = open(optfile, "r")
  s        = split(readline(fp))     #  QNSIZE, QNFREQ
  D.QNSIZE = toi(s[1])
  D.QNFREQ = toi(s[2])
  D.QSIZE  = zeros(Float64, D.QNSIZE)
  D.QFREQ  = zeros(Float64, D.QNFREQ)
  D.OPT    = zeros(Float64, D.QNSIZE, D.QNFREQ, 4)
  # the rest of the file    size + skip + {QNFREQ lines}
  for isize in 1:D.QNSIZE
    D.QSIZE[isize]  = tof(split(readline(fp))[1]) * 1.0e-4
    readline(fp)
    for ifreq in 1:D.QNFREQ
      s                       =  split(readline(fp))
      D.QFREQ[ifreq]          =  tof(s[1])   # keep QFREQ also as a separate array
      D.OPT[isize, ifreq, 1]  =  tof(s[1])   # freq
      D.OPT[isize, ifreq, 2]  =  tof(s[2])   # Qabs
      D.OPT[isize, ifreq, 3]  =  tof(s[3])   # Qsca
      D.OPT[isize, ifreq, 4]  =  tof(s[4])   # g
    end
  end
end



function read_GSET_heat(D, cfile)
  """
  Read enthalpy information.
  TRUST file, fourth line:    TMIN, TMAX, NT, bulk density
  The remaining lines:        T,   E=erg/g,    C=erg/g/K
  Note: Dustem had data separately for different grain sizes (with different C??).
  We (unnecessarily) use a size grid taken from optical data gridding.  
  // File format:
  //    NSIZE
  //    { SIZE [um] }
  //    NT
  //    { T }  = list of temperature values [K]
  //    each SIZE = one row with E(T) values (cgs units)  
  """
  fp = open(cfile, "r")
  l = readline(fp)
  while(l[1]=='#')
    l = readline(fp)
  end
  D.CNSIZE =  toi(split(l)[1])  # number of sizes
  D.CSIZE  =  zeros(Float64,D.CNSIZE)
  for i in 1:D.CNSIZE
    l          = readline(fp)
    D.CSIZE[i] = tof(split(l)[1]) * 1.0e-4   # file has [um]
  end
  l      =  readline(fp)
  D.CNT  =  toi(split(l)[1])
  D.CT   =  zeros(Float64, D.CNT)
  for i in 1:D.CNT
    s        =  split(readline(fp))
    D.CT[i]  =  tof(s[1])
  end
  # we skip D.CC, directly reading D.CE[D.CNT, D.CNSIZE]
  #  note: the file is in different order, one row is one size, temperature runs over columns
  D.CE = zeros(Float64, D.CNT, D.CNSIZE)
  for isize in 1:D.CNSIZE
    s    =  split(readline(fp))
    vol  =  (4.0/3.0)*pi*(D.CSIZE[isize])^3.0
    for iT in 1:D.CNT
      D.CE[iT, isize] = tof(s[iT]) / vol    # CONVERTED TO Erg/cm3
    end
  end  
  D.CC = 1e30*ones(Float64,2,2)  # ok not to have CC?
  # We need to interpolate CE onto E, for the finals SIZE_A, NE grid
  # Set up energy bins ---  E[NEPO, NSIZE], where size is D.NSIZE and D.SIZE_A
  # When we read sizes, that specified not only SIZE_A but also Tmin, Tmax for each size???
  # CRT / GSETDustO uses:
  #            *  Tmin, Tmax from the size file 
  #            *  emin   = TemperatureToEnergy_Int(s, TMIN[s]) ;
  #               emax   = TemperatureToEnergy_Int(s, TMAX[s]) ;
  #            * E[s*NEPO+ie] =  emin+ie*(emax-emin)/(NEPO-1.0) ;  
  D.E = zeros(Float64, D.NEPO, D.NSIZE)  # --- values at bin borders
  for isize in 1:D.NSIZE
    emin   =  T2E(D, D.SIZE_A[isize], D.TMIN[isize])   # using CT, CE  -->   [erg]
    emax   =  T2E(D, D.SIZE_A[isize], D.TMAX[isize])
    # @printf("TMIN %10.3e EMIN %12.3e    TMAX %10.3e EMAX %10.3e   size %.3e\n", D.TMIN[isize], emin, D.TMAX[isize], emax, D.SIZE_A[isize]*1e4)
    for ie  in 1:D.NEPO      
      ## D.E[ie, isize] = exp(log(emin)+(ie/D.NE)*(log(emax)-log(emin)))
      ## D.E[ie, isize] =  emin + (ie-1.0) * (emax-emin)/(D.NEPO-1.0)  #  [erg]
      # This is how GSETDustO does it -- linear in temperature
      D.E[ie, isize] =  T2E(D, D.SIZE_A[isize], 
      D.TMIN[isize] + (D.TMAX[isize]-D.TMIN[isize]) * ((ie-1.0)/(D.NEPO-1.0))^2 )
    end
    # @printf(" size %3d --- TMIN %7.2f   TMAX %7.2f\n", isize, D.TMIN[isize], D.TMAX[isize])    
  end
  
  if (false)
    println()
    @printf("SIZE %.3e TMIN %10.3e GIVES EMIN %10.3e\n", D.SIZE_A[1], D.TMIN[1], T2E(D, D.SIZE_A[1], D.TMIN[1]))
    for i = 1:D.CNT
      vol = (4.0/3.0)*pi*D.SIZE_A[1]^3
      @printf("   %12.4e %12.4e   %12.4e\n", D.CT[i], D.CE[i,1]*vol, T2E(D, D.SIZE_A[1], D.CT[i]))
    end
    # exit(0)
  end
  
end



############################################################################################################

  


  
function T2E(D, a, temp)
  """
  Convert temperature to energy for grains of size a. We use C.CT  and C.CE[CNT, CNSIZE].
  Note that D.CE is the energy per cm3 !!
  """
  # First find the closest size bins from CSIZE
  i = argmin(abs.(D.CSIZE.-a))
  if (a<D.CSIZE[i])
    i -= 1  # i should be the size below
  end
  j  = i + 1
  i  = clamp(i, 1, D.CNSIZE)
  j  = clamp(j, 1, D.CNSIZE)
  wi = 0.5
  if (i!=j)
    wi = (D.CSIZE[j]-a)/(D.CSIZE[j]-D.CSIZE[i])
  end
  # 
  if (true)
    Ei  = LIP(D.CT, D.CE[:,i], temp)                 # CE is erg/cm3 !!!
    Ej  = LIP(D.CT, D.CE[:,j], temp)
  else
    # linear interpolation on log scale
    Ei  = exp(LIP(log.(D.CT), log.(D.CE[:,i]), log(temp)))                 # CE is erg/cm3 !!!
    Ej  = exp(LIP(log.(D.CT), log.(D.CE[:,j]), log(temp)))
  end
  return (wi*Ei+(1.0-wi)*Ej) * (4.0/3.0)*pi*a^3.0  # only enthalpy erg/cm3 is interpolated
end
  


function E2T(D, a::Real, energy)
  """
  Convert energy erg of grains with size a to temperature.
  We use C.CT  and C.CE[CNT, CNSIZE] and linear interpolation in size.
  Note: C.CE is in [erg/cm3], argument is energy [erg].
  """
  # First find the closest size bins from CSIZE
  i = argmin(abs.(D.CSIZE.-a))
  if (a<D.CSIZE[i])
    i -= 1  # i should be the size below
  end
  j  = i + 1
  i  = clamp(i, 1, D.CNSIZE)
  j  = clamp(j, 1, D.CNSIZE)
  wj = 0.5
  if (i!=j)
    wj = (a-D.CSIZE[i])/(D.CSIZE[j]-D.CSIZE[i])
  end
  #
  vol = (4.0/3.0)*pi*a^3
  if (true)
    Ti  = LIP(D.CE[:,i], D.CT, energy/vol)            # note: D.CE is energy erg/cm3  ***NOT*** erg
    Tj  = LIP(D.CE[:,j], D.CT, energy/vol)
  else
    # Linear interpolation on log-log scale
    Ti  = exp(LIP(log.(D.CE[:,i]), log.(D.CT), log(energy/vol)))  # note: D.CE is energy erg/cm3 NOT erg
    Tj  = exp(LIP(log.(D.CE[:,j]), log.(D.CT), log(energy/vol)))
  end
  return ((1.0-wj)*Ti+wj*Tj)
end
  



function E2T(D, isize::Int, energy)
  """
  Convert energy erg of grains with size a to temperature.
  We use C.CT  and C.CE[CNT, CNSIZE] and linear interpolation in size.
  Note: C.CE is in [erg/cm3], argument is energy [erg].
  """
  return E2T(D, D.SIZE_A[isize], energy)
end
  
  
  
  
function write_simple_dust(DDS, filename)
  """
  Write dust description for CRT and SOC, based on DustEM but
  not assuming that DustEM will be used to calculate emission.
  This is a simple dust definition [freq, Qabs, Qsca, g] that is sufficient 
  for the radiative transfer part of the calculation.
  Input:
    DDS       = array of DData structures
    filename  = name of the written dust file
  Output:
    tmp.dust  = simple dust definition sufficient for radiative 
                transport part of the calculation with CRT or with SOC
  """
  println("write_simple_dust\n")
  freq  = DDS[1].FREQ
  NFREQ = length(freq)
  NDUST = length(DDS)            # number of dusts
  ABS   = zeros(NFREQ)           # total absorption cross section
  SCA   = zeros(NFREQ)
  GSUM  = zeros(NFREQ)
  for i in 1:NDUST
    Abs   =  Kabs(DDS[i], freq)  # interpolated values for each frequency in FREQ
    Sca   =  Ksca(DDS[i], freq)
    g     =  Gsca(DDS[i], freq)
    ABS  +=  Abs
    SCA  +=  Sca
    GSUM +=  Sca .* g
  end
  G             = GSUM ./ (SCA .+ 1.0e-40)
  GRAIN_DENSITY = 1.0e-7 
  GRAIN_SIZE    = 1.0e-4
  K             = GRAIN_DENSITY * pi*GRAIN_SIZE^2.0  # CRT will multipy Q factors with these
  fp            = open(filename, "w")
  @printf(fp, "eqdust\n")
  @printf(fp, "%12.5e\n", GRAIN_DENSITY)
  @printf(fp, "%12.5e\n", GRAIN_SIZE)
  @printf(fp, "%d\n",     NFREQ)
  for i in 1:(NFREQ)
    # we have (G, ABS, SCA) computed for new frequencies NFREQ
    @printf(fp, "%12.5e  %8.5f  %12.5e %12.5e\n", freq[i], G[i], ABS[i]/K, SCA[i]/K)
  end
  close(fp)
end


function HenyeyGreenstein(theta, g)
  # Probability per solid angle (not per dtheta!)
  # integral [0,pi] [ HenyeyGreenstein x sin(theta) ] dtheta dphi == 1.0
  p = (1.0/(4.0*pi)) .* (1.0.-g.*g) ./ (1.0.+g.*g.-2.0.*g.*cos.(theta)).^1.5
  return p
end


function HG_per_theta(theta, g)
  # Return probability per dtheta according to Henyey-Greenstein.
  # Integral over[0,pi] HG_per_theta * dtheta  == 1.0
  return 2.0*pi .* sin.(theta) .* HenyeyGreenstein(theta, g)  
end


function DSF2_simple(D::DData, freq, cos_theta, sin_weight)
  # Return discretised scattering function {theta, SF}.
  # Calculate Kscat-weighted average scattering function over all sizes
  # note -- this is SF(theta), i.e. per dtheta and not per solid angle.
  ##    DSF2_simple(cos_theta)                                   
  ##    combined_scattering_function2_simple(cos_theta)   ....   ~ DSF2_simple
  ##    write dsc :  cos_theta = linspace(-1.0, 1.0, BINS)       ~ must be >=0 !!!
  ##    cos_theta increasing, theta decreasing!!!
  total      = zeros(Float64, length(cos_theta))
  g          = Gsca(D, freq)
  if (sin_weight)   # SF per theta
    total  = HG_per_theta(acos.(cos_theta), g)     # sin(theta) IS included !!
  else              # SF per solid angle
    total  = HenyeyGreenstein(acos.(cos_theta), g) # sin(theta) NOT included !!
  end
  if (true)   # double check normalisation
    theta = acos.(cos_theta)   # decreasing order
    I     = 0.0                # THETA IS IN DECREASING ORDER --- INTEGRAL WILL BE NEGATIVE
    if (sin_weight)      
      I     =           LIN(theta, total, 0.0, pi)
    else
      I     =  2.0*pi * LIN(theta, total.*sin.(theta), 0.0, pi)
    end
    total *= 1.0/abs(I)   # abs because theta axis was decreasing
    # @printf("DSF2_simple, renormalisation *= %10.3e (no sin_weight)\n", 1.0/abs(I))
  end
  return total        
end





function DSF_simple(D::DData, freq, theta, sin_weight)
  #  Return scattering function for this dust, given wavelength
  #    theta is in increasing order! (important only for the normalisation check....)
  total  =  zeros(Float64, length(theta))
  g      =  Gsca(D, freq)               # this is weighted average over sizes 
  if (sin_weight)
    total = HG_per_theta(theta, g)      # sin(theta) IS included !!
  else
    total = HenyeyGreenstein(theta, g)  # sin(theta) NOT included !!
  end
  if (true)
    # double check normalisation
    I  = 0.0
    if (sin_weight)
      I      =           LIN(theta, total, 0.0, pi)
    else
      I      =  2.0*pi * LIN(theta, total.*sin(theta), 0.0, pi)
    end
    total *= 1.0/abs(I)  # if theta were in decreasing order, I -> -I ...
    # @printf("DSF_simple  -- renormalisation *= %10.3e  (sin_weight=%d)\n", 1.0/abs(I), sin_weight)
  end
  return total        
end



function combined_scattering_function_simple(DD::Array{DData,1}, um, theta, sin_weight; size_sub_bins=500)
  # Combined scattering function for a list of dusts.
  #  Called with theta in increasing order.
  n    = length(DD)
  f    = um2f(um)
  sca  = zeros(Float64, n)
  res  = zeros(Float64, length(theta))
  W    = 0.0
  for i in 1:n
    w     =  Ksca(DD[i], f)   # Ksca = weight factor
    W    +=  w
    res  +=  w * DSF_simple(DD[i], f, theta, sin_weight)
  end
  res /= W
  return res
end
  


function combined_scattering_function2_simple(DDS, um, cos_theta; sin_weight=false)
  n    =  length(DDS)
  f    =  um2f(um)
  sca  =  zeros(Float64, n)
  res  =  zeros(Float64, length(cos_theta))
  W    =  0.0
  for i in 1:n
    w     =  Ksca(DDS[i], f)   # Ksca = weight factor, scalar
    W    +=  w
    res  +=  w  .*  DSF2_simple(DDS[i], f, cos_theta, sin_weight)
  end
  res  /= W     # must have proper normalisation (not only ratios)
  return res    
end



function SFlookupCT(DDS::Array{DData,1}, um, bins; sin_weight=true, size_sub_bins=500)
  #=
  Make a look-up table with N elements, u*N maps to theta distribution  
  Note:
     For packet generation, sin_weight must be true
     For packet weighting, sin_weight must be false
  Note:
     return cos(theta)
  =#
  theta    =  linspace(0, pi, 5*bins)    # internally higher bdiscretisation over theta
  Y        =  combined_scattering_function_simple(DDS, um, theta, sin_weight, size_sub_bins=size_sub_bins)
  P        =  cumsum(Y) + 1.0e-7*cumsum(ones(Float64, length(Y)))
  P      .-=  P[1]
  P      ./=  P[end]              # [0,1]
  P[1]     = -1.0e-7
  P[end]   =  1.0+1.0e-7
  res      =  LIP(P, cos(theta), linspace(0.0, 1.0, bins))
  return res  # theta values
  
end



function SFlookupCT_simple(DD::Array{DData}, um, bins; sin_weight=true)
  #=
  Make a look-up table with N elements, u*N maps to theta distribution
  Note:
  For packet generation, SIN_WEIGHT must be true
  For packet weighting, SIN_WEIGHT must be false
  Note:
  return cos(theta)
  =#
  theta    =  linspace(0, pi, 5*bins)
  Y        =  combined_scattering_function_simple(DD, um, theta, sin_weight)
  P        =  cumsum(Y) + 1e-7*cumsum(ones(Float64, length(Y)))
  P      .-=  P[1]
  P      ./=  P[end]                 # [0,1]
  P[1]     = -1.0e-7
  P[end]   =  1.0+1.0e-7
  res      =  LIP(P, cos.(theta), linspace(0.0, 1.0, bins))
  return res
end



function write_scattering_function(DDS::Array{DData,1}, dscfile)
  #   Write the dsc file for RT programmes.
  freq      = DDS[1].FREQ
  bins      = DDS[1].BINS
  cos_theta = linspace(-1.0, 1.0, bins)
  fp        = open(dscfile, "w")
  for ifreq in 1:length(freq)              # all wavelengths!
    X   =  combined_scattering_function2_simple(DDS, f2um(freq[ifreq]), cos_theta, sin_weight=false)
    X   =  clamp.(X, 1e-5*maximum(X), 1e20) # should not have zeros...
    write(fp, Array{Float32}(X))
  end
  for ifreq in 1:length(freq)
    X   =  SFlookupCT_simple(DDS, f2um(freq[ifreq]), bins, sin_weight=true)
    write(fp, Array{Float32}(X))
  end
  close(fp)
end




function Kabs(D::DData, freq::Real)
  """
  Return total absorption cross section (integrated over size distribution) 
  at the given frequency.
  This uses summation over log-spaced size grid (aka KabsCRT!)
  """
  # Find the frequency indices --- remember: D.QFREQ may be in decreasing order of frequency
  sig = Int64(sign(D.QFREQ[2]-D.QFREQ[1]))
  i   = argmin(abs.(D.QFREQ.-freq))
  if (D.QFREQ[i]>freq)
    i -= sig     # depends on whether freq is increasing or decreasing, i will be the lower frequency
  end
  j  =  i+1                     # larger index, smaller frequency
  #  the frequency   QFREQ[j] < freq < QFREQ[i]
  i  =  clamp(i, 1, D.QNFREQ)
  j  =  clamp(j, 1, D.QNFREQ)
  wj =  0.5
  if (i!=j)
    wj  = (freq-D.QFREQ[i])  / (D.QFREQ[j]-D.QFREQ[i])
  end
  # To be consistent with CRT (and DustEm), interpolate Q before scaling with a^2
  #           OPT[isize, ifreq, 4]
  y1    =  LIP(D.QSIZE, D.OPT[:,i,2], D.SIZE_A)
  y2    =  LIP(D.QSIZE, D.OPT[:,j,2], D.SIZE_A)
  y     =  (1.0-wj)*y1 + wj*y2
  kabs  =  pi * sum(D.S_FRAC .* y .* D.SIZE_A.^2.0 ) # S_FRAC = grains per bin / H
  return   kabs
end  



function SKabs(D::DData, isize::Int64, freq::Real)
  """
  Return Absorption cross section for a single size bin
  Q*pi*a^2 * S_FRAC
  """
  sig = Int64(sign(D.QFREQ[2]-D.QFREQ[1]))
  i   = argmin(abs.(D.QFREQ.-freq))
  if (D.QFREQ[i]>freq)
    i -= sig
  end
  j  =  i+1                     # larger index, smaller frequency
  i  =  clamp(i, 1, D.QNFREQ)
  j  =  clamp(j, 1, D.QNFREQ)
  wj =  0.5
  if (i!=j)
    wj  = (freq-D.QFREQ[i])  / (D.QFREQ[j]-D.QFREQ[i])
  end
  y1    =  LIP(D.QSIZE, D.OPT[:,i,2], D.SIZE_A[isize])   # OPT[isize, ifreq, 2]
  y2    =  LIP(D.QSIZE, D.OPT[:,j,2], D.SIZE_A[isize])
  q     =  (1.0-wj)*y1 + wj*y2                          # interpolated absorption cross section
  return   pi * D.S_FRAC[isize] * q * D.SIZE_A[isize]^2.0
end  



function Ksca(D::DData, freq::Real)
  """
  Return total scattering cross section (integrated over size distribution) 
  at the given frequency.
  This uses summation over log-sapced size grid (aka KscaCRT!)
  ALTERNATIVE WOULD BE TO DO INTEGRATION OVER THE ORIGINAL D.QSIZE GRID
  """
  # Find the frequency indices --- remember: D.QFREQ is in decreasing order of frequency
  sig = Int64(sign(D.QFREQ[2]-D.QFREQ[1]))
  i   = argmin(abs.(D.QFREQ.-freq))
  if (D.QFREQ[i]>freq)
    i -= sig
  end
  j  =  i+1                     # larger index, smaller frequency
  #  the frequency   QFREQ[j] < freq < QFREQ[i]
  i  =  clamp(i, 1, D.QNFREQ)
  j  =  clamp(j, 1, D.QNFREQ)
  wj =  0.5
  if (i!=j)
    wj  = (freq-D.QFREQ[i])  / (D.QFREQ[j]-D.QFREQ[i])
  end
  # To be consistent with CRT (and DustEm), interpolate Q before scaling with a^2
  #  OPT[QNSIZE, QNFREQ, 4]
  y1    =  LIP(D.QSIZE, D.OPT[:,i,3], D.SIZE_A)
  y2    =  LIP(D.QSIZE, D.OPT[:,j,3], D.SIZE_A)
  y     =  (1.0-wj)*y1 + wj*y2
  ksca  =  pi * sum(D.S_FRAC .* y .* D.SIZE_A.^2.0 )
  return   ksca
end  




function Gsca(D::DData, freq::Real)
  """
  Return effective g calculated as <g*Ksca> over the size distribution.
  Int(g*Ksca*a^2) / Int(Ksca*a^2)
  THIS VERSION FREE OF SIZE_F, USES DIRECTLY S_FRAC.
  """
  # println("Gsca()")
  sig = Int64(sign(D.QFREQ[2]-D.QFREQ[1]))
  i   = argmin(abs.(D.QFREQ.-freq))
  if (D.QFREQ[i]>freq)
    i -= sig
  end
  j  =  i+1                     # larger index, smaller frequency! --  QFREQ[j] < freq < QFREQ[i]
  i  =  clamp(i, 1, D.QNFREQ)
  j  =  clamp(j, 1, D.QNFREQ)
  wj =  0.5
  if (i!=j)
    wj  = (freq-D.QFREQ[i])  / (D.QFREQ[j]-D.QFREQ[i])
  end
  wi = 1.0-wj
  #   sum(  S_FRAC * Ksca * a^2 * g ) / sum( S_FRAC * Ksca * a^2 )
  # Need to interpolate OPT from QSIZE to SIZE_A .... OPT[QSIZE, QFREQ, 4] = [ um, Kabs, Ksca, g ]
  
  W     =  D.S_FRAC .* D.SIZE_A.^2                 #  S_FRAC * a^2
  k     =  LIP(D.QSIZE, D.OPT[:, i, 3], D.SIZE_A)  #  Ksca
  g     =  LIP(D.QSIZE, D.OPT[:, i, 4], D.SIZE_A)  #  g
  gi    =  sum(W .* k .* g) / sum(W .* k)          # <g> at frequency i
  # The same for the second frequency
  k     =  LIP(D.QSIZE, D.OPT[:, j, 3], D.SIZE_A)  #  Ksca
  g     =  LIP(D.QSIZE, D.OPT[:, j, 4], D.SIZE_A)  #  g
  gj    =  sum(W .* k .* g) / sum(W .* k)          # <g> at frequency i
  #
  return  (wi*gi+(1.0-wi)*gj)
end  



function Kabs(D::DData, freq::Array{<:Real,1})::Array{Float64,1}
  res = zeros(Float64, length(freq))
  for i in 1:length(freq)
    res[i] = Kabs(D, freq[i])
  end
  return res
end


function Ksca(D::DData, freq::Array{<:Real,1})::Array{Float64,1}
  res = zeros(Float64, length(freq))
  for i in 1:length(freq)
    res[i] = Ksca(D, freq[i])
  end
  return res
end


function Gsca(D::DData, freq::Array{<:Real,1})::Array{Float64,1}
  res = zeros(Float64, length(freq))
  for i in 1:length(freq)
    res[i] = Gsca(D, freq[i])
  end
  return res
end



function set_frequencies(D::DData, freq::Array{Float64,1})
  D.FREQ = freq
  D.Ef   = PLANCK*freq
end



function PrepareTdownCL(D::DData)
  #=
  Using the same kernel as A2E_pre.py (which is known to work).
  Returns Tdown{Float64,2}[D.NE, D.NSIZE]
  =#  
  device, ctx, queue = cl.create_compute_context()  
  src     =  read(open(homedir()*"/starformation/SOC/kernel_A2E_pre.c"), String)
  prg0    =  cl.Program(ctx, source=src)
  opts    =  @sprintf("-D FACTOR=%.4ef", FACTOR)
  prg     =  cl.build!(prg0, options=opts) # input column-major
  kernel  =  cl.Kernel(prg, "PrepareTdown")
  LOCAL   =  Int32(5)
  GLOBAL  =  (Int32(floor(D.NE/64))+1)*64
  TDOWN   =  zeros(Float64, D.NE, D.NSIZE)
  # Inputs for the kernel
  NFREQ   =  Int32(length(D.FREQ))
  FREQ    =  Array{Float32}(D.FREQ)
  Ef      =  Array{Float32}(PLANCK*FREQ)
  NE      =  Int32(D.NE)
  FREQ_buf  = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=FREQ)
  Ef_buf    = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=Ef)
  Tdown_buf = cl.Buffer(Float32, ctx, :w,          NE)
  # Buffers that change for each size
  SKABS     =  zeros(Float32, NFREQ)
  SKABS_buf =  cl.Buffer(Float32, ctx, :r, NFREQ)     # NFREQ = one size
  E_buf     =  cl.Buffer(Float32, ctx, :r, D.NEPO)
  T_buf     =  cl.Buffer(Float32, ctx, :r, D.NEPO)
  for isize=1:D.NSIZE
    for ifreq=1:NFREQ
      # SKABS in kernel == Q*pi*a^2
      SKABS[ifreq] =  SKabs(D, isize, FREQ[ifreq]) / D.S_FRAC[isize]
    end
    E       =  Array{Float32}(D.E[:, isize])
    T       =  Array{Float32}(E2T(D, isize, E))
    cl.write!(queue, SKABS_buf, SKABS)
    cl.write!(queue, E_buf,     E)
    cl.write!(queue, T_buf,     T)
    queue(kernel, GLOBAL, LOCAL, NFREQ, FREQ_buf, Ef_buf, SKABS_buf, NE, E_buf, T_buf, Tdown_buf)
    tdown = cl.read(queue, Tdown_buf)
    TDOWN[:, isize] = tdown
  end
  return TDOWN
end



function PrepareTdown(D::DData)
  """
  Prepare vector TDOWN = transition matrix elements for transitions u -> u-1
  D&L01 Eq. 41
  """
    
  TDOWN    =  zeros(Float64, D.NE, D.NSIZE)  # NE runs faster
  i        =  0
  Ef       =  D.Ef
  Eu, El, Tu = 0.0, 0.0, 0.0
  NFREQ    =  length(D.FREQ)
  ee0      =  0.0
  ee1      =  0.0
  yy0      =  0.0
  I        =  0.0
  factor   =  8.0*pi/C_LIGHT^2
  
  # julia is column order, keep files such the NEPO runs faster -> E[NEPO, NSIZE]
  for isize in 1:D.NSIZE
    # vol    =  (4.0/3.0)*pi*(D.SIZE_A[size]^3)
    # We have D.CE as Erg/cm3 ----- but D.E[] is in  Erg
    # In GSETDustO this is directly values read from the file !!!!!
    e      =  D.E[:, isize]            # in E[isize*NEPO+ie] .... in julia E[ie, isize]
    TDOWN[1,isize] = 0.0               # in julia TDOWN[NE, NSIZE]
    for u in 2:D.NE      
      Eu   =  Float64(0.5*(e[u]+e[u+1]))        #  e[NEPO],  E[NEPO, NSIZE]
      El   =  Float64(0.5*(e[u-1]+e[u]))      
      Tu   =  E2T(D, D.SIZE_A[isize], Eu)      
      # @printf("s=%d u=%d  Eu %10.3e  El %10.3e  Tu %10.3e\n", isize, u, Eu, El, Tu)
      yy1  =  0.0
      i    =  1
      I    =  0.0
      
      while ((i<NFREQ) && (Ef[i+1]<Eu))   # upper bin borders?
        ee0  =  Ef[i] ;
        yy0  =  ee0^3 *   SKabs(D, isize, ee0/PLANCK) / (exp(ee0/(BOLZMANN*Tu))-1.0) 
        for j in 1:8
          ee1  =  Ef[i] + j*(Ef[i+1]-Ef[i])/8.0 
          yy1  =  ee1^3 * SKabs(D, isize, ee1/PLANCK) / (exp(ee1/(BOLZMANN*Tu))-1.0) 
          I   +=  0.5*(ee1-ee0)*(yy1+yy0) 
          ee0  =  ee1 
          yy0  =  yy1           
        end
        i += 1
      end        
      
      if (Eu<Ef[NFREQ])
        for j = 1:8
          ee1 = Ef[i] + j*(Eu-Ef[i])/8.0
          yy1 = ee1^3 * SKabs(D, isize, ee1/PLANCK) / (exp(ee1/(BOLZMANN*Tu))-1.0)
          I  += 0.5*(ee1-ee0)*(yy1+yy0)
          ee0 = ee1
          yy0 = yy1
        end
      end
      
      # Note:  D.S_FRAC   ==  GRAIN_DENSITY * S_GFRAC in c-version
      I  *=  8.0*pi/((Eu-El)*C_LIGHT^2*PLANCK^3 *  D.S_FRAC[isize])
      TDOWN[u, isize] = I

      if (false)
        @printf("size %7.4f um  u=%d  Tu=%7.4fK E=%12.4e->%12.4e  I=%10.3e Tdown=%10.3e\n",
        1.0e4*D.SIZE_A[isize], u, Tu,   Eu,     El,     I,             TDOWN[u, isize]) ;
        @printf("GRAIN_DENSITY %.3e,  S_FRAC[0] %.3e\n", D.GRAIN_DENSITY, D.S_FRAC[1]) ;
        exit(0)
      end        

      if (false)
        if (isize==1)
          # Dump the Tdown vector
          fp = open("jTD.dump", "w")
          @printf(fp, "# a[1] = %.3e\n", D.SIZE_A[isize])
          for ie in 1:D.NE
            @printf(fp, "%12.4e\n", TDOWN[ie, isize])
          end
          close(fp)
        end
      end

      
    end # upper bin
  end # isize
  return  TDOWN
end




function PrepareIntegrationWeights(D::DData, isize)
  """
  Precalculate integration weights for upwards transitions.
  Eq. 15, (16) in Draine&Li (2001)
  With the weight precalculated, the (trapezoidal numerical) integral of eq. 15 is a dot product of the vectors
  containing the weights and the absorbed photons (=Cabs*u), which can be calculated
  very quickly with (fused) multiply-add.
  """
  NFREQ            = length(D.FREQ)
  weights_for_size = 0
  L1               = zeros(Int32, (D.NE, D.NE))   # single size --->   D.L1[:,:,isize], L1 = L1[u,l]
  L2               = zeros(Int32, (D.NE, D.NE))
  n_int_points     = 0
  freq_e2          = zeros(Float64, NFREQ+4)
  Iw               = zeros(Float64, NFREQ, D.NE, D.NE)   # julia [ifreq, u, l]
  e                = view(D.E, :, isize)                 # julia D.E[D.NEPO, D.NSIZE]
  W                = zeros(Float64, 4)
  Eu               = 0.0
  El               = 0.0
  Ef               = D.Ef
  
  for l in 1:(D.NE-1)
    dEl =  e[l+1] - e[l]
    El  =  0.5*(e[l]+e[l+1])
    
    
    # @printf("isize %d, dEl %12.4e, El %12.4e\n", isize, dEl, El)
    
    
    for u in (l+1):D.NE        
      Eu    = 0.5*(e[u]+e[u+1])
      dEu   = e[u+1]-e[u]
      W[1]  = e[u]-e[l+1]
      W[2]  = min( e[u]-e[l],  e[u+1]-e[l+1] )
      W[3]  = max( e[u]-e[l],  e[u+1]-e[l+1] )
      W[4]  = e[u+1] - e[l]
      
      if (Ef[1]>W[4] || Ef[NFREQ]<W[1])
        # in c  L1[l*NE+u] .... in julia L1[u,l]
        L1[u, l] = -1
        L2[u, l] = -2
        continue
      end
      
      i, j, k = 1, 1, 1   # julia += 1
      while (i<=(NFREQ+4))
        # after this freq_e2 contains all the breakpoints of the integrand: 
        # NFREQ from freq grids + 4 from bin edges
        if (j<=NFREQ && k<=4)
          freq_e2[i]  =  min(Ef[j], W[k])
          if (Ef[j]<W[k]) 
            j += 1 
          else 
            k += 1
          end
        else
          if (j>NFREQ && k<=4) 
            freq_e2[i] = W[k]
          elseif (k>4 && j<=NFREQ) 
            freq_e2[i]=Ef[j]
          else 
            assert(1==0)
          end
        end
        i += 1
      end # while
      
      # @printf("ijk %2d %2d %2d    Eu %12.4e  dEu %12.4e   W %10.3e %10.3e %10.3e %10.3e\n", i, j, k, Eu, dEu, W[1], W[2], W[3], W[4])
      

      # done: integrate Gul*nabs*E over [freq_e2[i-1], freq_e2[i]]
      # Gul, NABS and E are all linear functions on the interval,
      # so the integrand is (at most) third degree polynomial
      
      for i=2:(NFREQ+4) 


        # println(i-1)
        
        if (freq_e2[i-1]>Ef[NFREQ] || freq_e2[i-1]>=W[4]) # Ef[i] = photon energy at freq[i]
          break                          
        end
        if (freq_e2[i]<W[1] || freq_e2[i-1]<=Ef[1]) 
          continue
        end
        
        j = 1
        while (Ef[j]<freq_e2[i-1] && j<(NFREQ-1))
          j += 1
        end
        
        # *** i and j +1 wrt c version ***
        
        # 1
        if (W[2]<=freq_e2[i-1] && freq_e2[i]<=W[3])  # on the flat part of Gul
          Gul = min(dEu, dEl)/(dEu*dEl) 
          
          if (freq_e2[i]<=Ef[j])                     # between Ef[j-1] and Ef[j]
            Iw[j,u,l] += Gul*(freq_e2[i]*freq_e2[i]*(1.0/3.0*freq_e2[i]-1.0/2.0*Ef[j-1]) -
            freq_e2[i-1]*freq_e2[i-1]*(1.0/3.0*freq_e2[i-1]-1.0/2.0*Ef[j-1])) / (Ef[j]-Ef[j-1]+1e-120) ;
            Iw[j-1,u,l] += Gul*(freq_e2[i]*freq_e2[i]*(-1.0/3.0*freq_e2[i]+1.0/2.0*Ef[j]) -
            freq_e2[i-1]*freq_e2[i-1]*(-1.0/3.0*freq_e2[i-1]+1.0/2.0*Ef[j])) / (Ef[j]-Ef[j-1]+1e-120) ;
          else                                       # between Ef[j] and Ef[j+1]
            Iw[j+1,u,l] += Gul*(freq_e2[i]*freq_e2[i]*(1.0/3.0*freq_e2[i]-1.0/2.0*Ef[j]) -
            freq_e2[i-1]*freq_e2[i-1]*(1.0/3.0*freq_e2[i-1]-1.0/2.0*Ef[j])) / (Ef[j+1]-Ef[j]+1e-120) ;
            Iw[j,u,l] += Gul*(freq_e2[i]*freq_e2[i]*(-1.0/3.0*freq_e2[i]+1.0/2.0*Ef[j+1]) -
            freq_e2[i-1]*freq_e2[i-1]*(-1.0/3.0*freq_e2[i-1]+1.0/2.0*Ef[j+1])) / (Ef[j+1]-Ef[j]+1e-120) ;
          end
        end


        #@printf("1: u=%d, l=%d, j=%d    Iw=%12.4e\n", u, l, j-1, Iw[j-1,u,l])
        #@printf("1: u=%d, l=%d, j=%d    Iw=%12.4e\n", u, l, j,   Iw[j  ,u,l])
        #@printf("1: u=%d, l=%d, j=%d    Iw=%12.4e\n", u, l, j+1, Iw[j+1,u,l])
        #@printf("1: u=%d, l=%d, j=%d    Iw=%12.4e\n", 2, 1, 1,   Iw[1,2,1])
        
        
        
        # 2
        if (W[1]<=freq_e2[i-1] && freq_e2[i]<=W[2])     # on the upslope
          
          if (freq_e2[i]<=Ef[j])  # between Ef[j-1] and Ef[j]
            
            Iw[j,u,l] += (freq_e2[i]*freq_e2[i]*(1.0/4.0*freq_e2[i]*freq_e2[i] -
            1.0/3.0*(Ef[j-1]+W[1])*freq_e2[i] + 0.5*W[1]*Ef[j-1]) -
            freq_e2[i-1]*freq_e2[i-1]*(1.0/4.0*freq_e2[i-1]*freq_e2[i-1] -
            1.0/3.0*(Ef[j-1]+W[1])*freq_e2[i-1] + 0.5*W[1]*Ef[j-1])) / (dEl*dEu*(Ef[j]-Ef[j-1]+1e-120))
            
            Iw[j-1,u,l] += (freq_e2[i]*freq_e2[i]*(-1.0/4.0*freq_e2[i]*freq_e2[i] +
            1.0/3.0*(Ef[j]+W[1])*freq_e2[i] - 0.5*W[1]*Ef[j]) -
            freq_e2[i-1]*freq_e2[i-1]*(-1.0/4.0*freq_e2[i-1]*freq_e2[i-1] +
            1.0/3.0*(Ef[j]+W[1])*freq_e2[i-1] - 0.5*W[1]*Ef[j])) / (dEl*dEu*(Ef[j]-Ef[j-1]+1e-120)) ;
            
          else # between Ef[j] and Ef[j+1]
            
            Iw[j+1,u,l] += (freq_e2[i]*freq_e2[i]*(1.0/4.0*freq_e2[i]*freq_e2[i] -
            1.0/3.0*(Ef[j]+W[1])*freq_e2[i] + 1.0/2.0*W[1]*Ef[j]) -
            freq_e2[i-1]*freq_e2[i-1]*(1.0/4.0*freq_e2[i-1]*freq_e2[i-1] -
            1.0/3.0*(Ef[j]+W[1])*freq_e2[i-1] + 1.0/2.0*W[1]*Ef[j])) / (dEl*dEu*(Ef[j+1]-Ef[j]+1e-120)) ;
            
            Iw[j,u,l] += (freq_e2[i]*freq_e2[i]*(-1.0/4.0*freq_e2[i]*freq_e2[i] +
            1.0/3.0*(Ef[j+1]+W[1])*freq_e2[i] - 1.0/2.0*W[1]*Ef[j+1]) -
            freq_e2[i-1]*freq_e2[i-1]*(-1.0/4.0*freq_e2[i-1]*freq_e2[i-1] +
            1.0/3.0*(Ef[j+1]+W[1])*freq_e2[i-1] - 1.0/2.0*W[1]*Ef[j+1])) / (dEl*dEu*(Ef[j+1]-Ef[j]+1e-120)) ;
            
          end
        end  

        
        #@printf("2: u=%d, l=%d, j=%d    Iw=%12.4e\n", u, l, j-1, Iw[j-1,u,l])
        #@printf("2: u=%d, l=%d, j=%d    Iw=%12.4e\n", u, l, j,   Iw[j  ,u,l])
        #@printf("2: u=%d, l=%d, j=%d    Iw=%12.4e\n", u, l, j+1, Iw[j+1,u,l])
        #@printf("2: u=%d, l=%d, j=%d    Iw=%12.4e\n", 2, 1, 1,   Iw[1,2,1])
        

        
        # 3              
        if (W[3]<=freq_e2[i-1] && freq_e2[i]<=W[4])         # on the downslope
          
          if (freq_e2[i]<=Ef[j])                        # between Ef[j-1] and Ef[j]
            
            Iw[j,u,l] += (freq_e2[i]*freq_e2[i]*(-1.0/4.0*freq_e2[i]*freq_e2[i] +
            1.0/3.0*(Ef[j-1]+W[4])*freq_e2[i] - 1.0/2.0*W[4]*Ef[j-1]) -
            freq_e2[i-1]*freq_e2[i-1]*(-1.0/4.0*freq_e2[i-1]*freq_e2[i-1] +
            1.0/3.0*(Ef[j-1]+W[4])*freq_e2[i-1] - 1.0/2.0*W[4]*Ef[j-1])) / (dEl*dEu*(Ef[j]-Ef[j-1]+1e-120)) ;
            
            Iw[j-1,u,l] += (freq_e2[i]*freq_e2[i]*(1.0/4.0*freq_e2[i]*freq_e2[i] -
            1.0/3.0*(Ef[j]+W[4])*freq_e2[i] + 1.0/2.0*W[4]*Ef[j]) -
            freq_e2[i-1]*freq_e2[i-1]*(1.0/4.0*freq_e2[i-1]*freq_e2[i-1] -
            1.0/3.0*(Ef[j]+W[4])*freq_e2[i-1] + 1.0/2.0*W[4]*Ef[j])) / (dEl*dEu*(Ef[j]-Ef[j-1]+1e-120)) ;
            
          else                                          # between Ef[j] and Ef[j+1]
            
            Iw[j+1,u,l] += (freq_e2[i]*freq_e2[i]*(-1.0/4.0*freq_e2[i]*freq_e2[i] +
            1.0/3.0*(Ef[j]+W[4])*freq_e2[i] - 1.0/2.0*W[4]*Ef[j]) -
            freq_e2[i-1]*freq_e2[i-1]*(-1.0/4.0*freq_e2[i-1]*freq_e2[i-1] +
            1.0/3.0*(Ef[j]+W[4])*freq_e2[i-1] - 1.0/2.0*W[4]*Ef[j])) / (dEl*dEu*(Ef[j+1]-Ef[j]+1e-120)) ;
            
            Iw[j,u,l] += (freq_e2[i]*freq_e2[i]*(1.0/4.0*freq_e2[i]*freq_e2[i] -
            1.0/3.0*(Ef[j+1]+W[4])*freq_e2[i] + 1.0/2.0*W[4]*Ef[j+1]) -
            freq_e2[i-1]*freq_e2[i-1]*(1.0/4.0*freq_e2[i-1]*freq_e2[i-1] -
            1.0/3.0*(Ef[j+1]+W[4])*freq_e2[i-1] + 1.0/2.0*W[4]*Ef[j+1])) / (dEl*dEu*(Ef[j+1]-Ef[j]+1e-120)) ;
            
          end
        end
        
        
        #@printf("3: u=%d, l=%d, j=%d    Iw=%12.4e\n", u, l, j-1, Iw[j-1,u,l])
        #@printf("3: u=%d, l=%d, j=%d    Iw=%12.4e\n", u, l, j,   Iw[j  ,u,l])
        #@printf("3: u=%d, l=%d, j=%d    Iw=%12.4e\n", u, l, j+1, Iw[j+1,u,l])
        #@printf("3: u=%d, l=%d, j=%d    Iw=%12.4e\n", 2, 1, 1,   Iw[1,2,1])
        
        
      end  # for NFREQ
      
      
     
      
      
      # intrabin absorptions: second term of Eq. 28                  
      # assume NABS piecewise linear and integrate the product of 2 or 3 linear functions...
      
      if (u==l+1) 
        j = 2
        
        while ((j<=NFREQ) && (Ef[j]<dEl))
          
          Iw[j,u,l] += (Ef[j]*Ef[j]*(1.0/3.0*Ef[j]-1.0/2.0*Ef[j-1]-1.0/4.0*Ef[j]*Ef[j]/dEl+1.0/3.0*Ef[j-1]*Ef[j]/dEl) -
          Ef[j-1]*Ef[j-1]*(1.0/3.0*Ef[j-1]-1.0/2.0*Ef[j-1]-1.0/4.0*Ef[j-1]*Ef[j-1]/dEl+1.0/3.0*Ef[j-1]*Ef[j-1]/dEl)) /
          (dEu*(Ef[j]-Ef[j-1]+1e-120))
          
          Iw[j-1,u,l] += (Ef[j]*Ef[j]*(-1.0/3.0*Ef[j]+1.0/2.0*Ef[j]+1.0/4.0*Ef[j]*Ef[j]/dEl-1.0/3.0*Ef[j]*Ef[j]/dEl) -
          Ef[j-1]*Ef[j-1]*(-1.0/3.0*Ef[j-1]+1.0/2.0*Ef[j]+1.0/4.0*Ef[j-1]*Ef[j-1]/dEl-1.0/3.0*Ef[j]*Ef[j-1]/dEl)) /
          (dEu*(Ef[j]-Ef[j-1]+1e-120))
          
          #@printf("IB  u=%d, l=%d, j=%d    Iw=%12.4e\n", u, l, j-1, Iw[j-1,u,l])
          #@printf("IB  u=%d, l=%d, j=%d    Iw=%12.4e\n", u, l, j,   Iw[j  ,u,l])
          
          j +=1
          
        end
        
        if (j<=NFREQ)
          
          Iw[j,u,l] += (dEl*dEl*(1.0/3.0*dEl-1.0/2.0*Ef[j-1]-1.0/4.0*dEl+1.0/3.0*Ef[j-1]) -
          Ef[j-1]*Ef[j-1]*(1.0/3.0*Ef[j-1]-1.0/2.0*Ef[j-1]-1.0/4.0*Ef[j-1]*Ef[j-1]/dEl+1.0/3.0*Ef[j-1]*Ef[j-1]/dEl))/
          (dEu*(Ef[j]-Ef[j-1]+1e-120))
          
          Iw[j-1,u,l] += (dEl*dEl*(1.0/2.0*Ef[j]-1.0/3.0*dEl-1.0/3.0*Ef[j]+1.0/4.0*dEl) -
          Ef[j-1]*Ef[j-1]*(1.0/2.0*Ef[j]-1.0/3.0*Ef[j]-1.0/3.0*Ef[j]*Ef[j]/dEl+1.0/4.0*Ef[j]*Ef[j]/dEl))/
          (dEu*(Ef[j]-Ef[j-1]+1e-120))
          
          #@printf("IB  u=%d, l=%d, j=%d    Iw=%12.4e\n", u, l, j-1, Iw[j-1,u,l])
          #@printf("IB  u=%d, l=%d, j=%d    Iw=%12.4e\n", u, l, j,   Iw[j  ,u,l])

          
        end
      end
      

      # @printf("IB: u=%d, l=%d, j=%d    Iw=%12.4e\n", 2, 1, 1,   Iw[1,2,1])
      
            
      first_non_zero = -1 
      last_non_zero  = -2              
      
      for i in 1:NFREQ          
        if (Iw[i,u,l]>0.0 && first_non_zero<1) 
          first_non_zero = i
        end
        if (Iw[i,u,l]>0.0) 
          last_non_zero = i
        end
      end
      
      L1[u, l] = first_non_zero ;
      L2[u, l] = last_non_zero ;
      
      for i in  L1[u,l]:L2[u,l]
        if (i<=NFREQ) 
          Iw[i,u,l] *=  dEu/((Eu-El)* (1.0e20*PLANCK))  # 1e-20 = SCALE in RT programs
          
          # @printf("u=%3d l=%3d i=%3d   %12.4e   %12.4e\n", u, l, i, Iw[i,u,l], dEu/((Eu-El)* (1.0e20*PLANCK)))
          
        else 
          Iw[i,u,l] = 0.0
        end
      end      
      
      # @printf("::: u=%3d l=%3d i=%3d   %12.4e\n", 2, 1, 1, Iw[1,2,1])
      
      # exit(0)
      
      
    end # for u
  end # for l


  if (false)
    # Dump separate file for isize=1 rates  ---- Iw[ifreq, u, l]
    #  all values l=[1,NE[,  u=]l,NE], NFREQ
    if (isize==1)
      xx = Float32(0.0)
      fp = open("jW.dump", "w")
      for l = 1:D.NE  # include zeros !!
        for u = 1:D.NE
          for i = 1:NFREQ
            xx = Float32(  Iw[i, u, l] )
            write(fp, xx)
          end
        end
      end
      close(fp )
    end
  end

  
  # L1, L2 [u,l]     = index of the first and the last frequency used in the integral
  # Iw[ifreq, u, l]  = the integration weights
  return L1, L2, Iw
  
end  #  PrepareIntegrationWeights(D::DData, isize)





# function DumpSolverData(DD::Array{DData,1}, filename, freq)
function DumpSolverData(D::DData, filename, freq, with_cl=false)
  """
  Dump data in the format needed by A2E_pyCL.py 
  ... so that we can avoid CRT/A2E altogether when using SOC + A2E_pyCL.
  
  2018-12-14 -- each dust in a separate file
  File format:
      NFREQ
      FREQ[NFREQ]
      GRAIN_DENSITY
      NSIZE
      S_FRAC[NSIZE]
      NE
      SK_ABS[NSIZE, NFREQ] --- frequency runs faster
      --- for each size ---
      
             noIw          --- number of weights THIS SIZE
             Iw[noIw]      --- nonzero weight THIS SIZE
             L1[NE,NE]     
             L2[NE,NE]
             TDOWN[NE]
             EA[NE, NFREQ]   -- frequency runs faster
             Ibeg[NFREQ]

  ---------------------------------------------------------------------------
  """
  println("DumpSolverData")
  nfreq  =  length(freq)
  fp     =  open(filename, "w")
  write(fp, Int32(nfreq))                                   #  NFREQ
  write(fp, Array{Float32}(freq))                           #  FREQ[NFREQ]
  write(fp, Float32(D.GRAIN_DENSITY))                       #  GRAIN_DENSITY
  write(fp, Int32(D.NSIZE))                                 #  NSIZE
  write(fp, Array{Float32}(D.S_FRAC/D.GRAIN_DENSITY))       #  S_FRAC[NSIZE]
  write(fp, Int32(D.NE))                                    #  NE
  #   SK_ABS
  tmp   =  zeros(Float32, nfreq, D.NSIZE)
  # @printf("SKabs %d x %d = %d elements\n", D.NSIZE, nfreq, D.NSIZE*nfreq)
  for isize = 1:D.NSIZE
    for ifreq = 1:nfreq
      tmp[ifreq, isize] = SKabs(D, isize, freq[ifreq])
    end
  end            
  write(fp, tmp)                                            # SKABS[nfreq, NSIZE]
  t0 = time()
  # Solve Tdown already for all grain sizes
  if with_cl
    Tdown =  PrepareTdownCL(D)                                # Tdown[NE, NSIZE]
  else
    Tdown =  PrepareTdown(D) 
  end
  @printf(" --- PrepareTdown: %.3f seconds\n", time()-t0)



  if with_cl
    device, ctx, queue = cl.create_compute_context()  
    src     =  read(open(homedir()*"/starformation/SOC/kernel_A2E_pre.c"), String)
    prg0    =  cl.Program(ctx, source=src)
    opts    =  @sprintf("-D FACTOR=%.4ef", FACTOR)
    prg     =  cl.build!(prg0, options=opts) # input column-major
    kernel  =  cl.Kernel(prg, "PrepareIntegrationWeights")
    LOCAL   =  Int32(4)
    GLOBAL  =  (Int32(floor(D.NE/64))+1)*64
    NFREQ   =  Int32(length(D.FREQ))
    NE      =  Int32(D.NE)
    Ef      =  Array{Float32}(PLANCK*D.FREQ)
    E       =  zeros(Float32, D.NEPO)
    T       =  zeros(Float32, D.NEPO)
    EA      =  zeros(Float32, NE, NFREQ)
    SKABS   =  zeros(Float32, NFREQ)
    Ibeg    =  zeros(Int32, NFREQ)
    Ef_buf  =  cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=Ef)
    #
    E_buf   =  cl.Buffer(Float32, ctx, :r,  D.NEPO)
    L1_buf  =  cl.Buffer(Int32,   ctx, :rw, NE*NE)
    L2_buf  =  cl.Buffer(Int32,   ctx, :rw, NE*NE)
    wrk_buf =  cl.Buffer(Float32, ctx, :rw, NE*NFREQ+NE*(NFREQ+4))
    noIw_buf=  cl.Buffer(Int32,   ctx, :rw, NE-1)
    Iw_buf  =  cl.Buffer(Float32, ctx, :rw, NE*NE*NFREQ)
    LOCAL   =  4
    GLOBAL  =  (Int32(floor(D.NE/64))+1)*64
    
    for isize=1:D.NSIZE      
      emin  =  Float32(T2E(D, D.SIZE_A[isize], D.TMIN[isize]))
      emax  =  Float32(T2E(D, D.SIZE_A[isize], D.TMAX[isize]))
      for ie=1:D.NEPO
        T[ie]  =  D.TMIN[isize] +  (D.TMAX[isize]-D.TMIN[isize]) * ((ie-1.0)/(D.NEPO-1.0))^2.0
        E[ie]  =  T2E(D, D.SIZE_A[isize], T[ie])
      end
      # @printf("isize=%2d  %.5f um   T %6.1f %6.1f   E %12.4e %12.4e\n", isize,  D.SIZE_A[isize], D.TMIN[isize], D.TMAX[isize], emin, emax)
      cl.write!(queue, E_buf, E)
      queue(kernel, GLOBAL, LOCAL, NFREQ, NE, Ef_buf, E_buf, L1_buf, L2_buf, Iw_buf, wrk_buf, noIw_buf)
      Iw        =  cl.read(queue, Iw_buf)
      noIw      =  cl.read(queue, noIw_buf)
      sum_noIw  =  sum(noIw)
      write(fp, Int32(sum_noIw))      
      for l=0:(NE-2)  # as in Python, index [0,NE-1[
        ind    =  l*NE*NFREQ
        write(fp, Array{Float32}( Iw[(ind+1):(ind+noIw[l+1])] ))
      end
      L      =  cl.read(queue, L1_buf)
      L[1]   = -2
      write(fp, L)
      L      =  cl.read(queue, L2_buf)
      L[1]   = -2
      write(fp, L)
      write(fp, Array{Float32}(Tdown[:,isize]))    #  Tdown[D.NE, D.NSIZE]
      # EA
      for ie in 1:D.NE     #  D.E[NEPO, NSIZE]
        # isize refers to discrete grain sizes in the SIZE_A array
        t  =  E2T(D, isize, 0.5*(D.E[ie,isize]+D.E[ie+1,isize]))  # [D.E] = erg
        for ifreq in 1:nfreq
          f = freq[ifreq]
          EA[ie,ifreq] = (4.0e20*pi/PLANCK) * SKabs(D,isize,f) * PlanckIntensity(f, t) / f
        end # for ifreq
      end # for ie
      write(fp, EA)
      # Ibeg
      for ifreq in 1:nfreq
        startind = 2
        # E[NEPO, NSIZE]
        while( (startind<D.NEPO) && (0.5*(D.E[startind-1,isize]+D.E[startind,isize])<D.Ef[ifreq]) )
          startind += 1
        end
        Ibeg[ifreq] = startind - 1 ; #  write as 0-offset indices !!
      end
      write(fp, Ibeg)
    end # for isize



    
  else  # else not OpenCL
    
    for isize in 1:D.NSIZE            
      # L1[u,l],   IW[ifreq, u, l]
      t0 = time()
      L1, L2, IW    =  PrepareIntegrationWeights(D, isize)
      # @printf(" --- PrepareIntegrationWeightsTdown: %.3f seconds\n", time()-t0)    
      # count the number of non-zero integration weights
      noIw = 0
      for l in 1:(D.NE-1)
        for u in (l+1):D.NE
          noIw +=  L2[u,l] - L1[u,l] +1
        end
      end    
      @printf("   noIw %6d\n", noIw)
      write(fp, Int32(noIw))                                  #  noIw
      # write nonzero weights --- C-order  size, l, u, ifreq
      for l=1:(D.NE-1)
        for u=(l+1):D.NE
          z = Array{Float32,1}( IW[ L1[u,l]:L2[u,l]  , u, l] )
          write(fp, z)                                        #  Iw[noIw]
        end
      end
      # write L1 and L2 **** AS 0-OFFSET VALUES
      write(fp, Array{Int32}(L1.-1))                          #  L1[NE*NE]
      write(fp, Array{Int32}(L2.-1))                          #  L2[NE*NE]
      # Tdown[NE, NSIZE]  --- NE for current size
      write(fp, Array{Float32}(Tdown[:,isize]))               #  Tdown[NE]
      # emission array -> in julia EMIT_ARRAY[NE, nfreq]
      EMIT_ARRAY  = zeros(Float32, D.NE, nfreq)
      for ie in 1:D.NE     #  D.E[NEPO, NSIZE]
        # isize refers to discrete grain sizes in the SIZE_A array
        T  =  E2T(D, isize, 0.5*(D.E[ie,isize]+D.E[ie+1,isize]))  # [D.E] = erg
        for ifreq in 1:nfreq
          f = freq[ifreq]
          EMIT_ARRAY[ie,ifreq] = (4.0e20*pi/PLANCK) * SKabs(D,isize,f) * PlanckIntensity(f, T) / f
        end # for t
      end # for ie
      write(fp, Array{Float32}(EMIT_ARRAY))                   #  EA[NE*NFREQ]
      II = zeros(Int32, nfreq)
      for ifreq in 1:nfreq
        startind = 2
        # E[NEPO, NSIZE]
        while( (startind<D.NEPO) && (0.5*(D.E[startind-1,isize]+D.E[startind,isize])<D.Ef[ifreq]) )
          startind += 1
        end
        II[ifreq] = startind - 1 ; #  write as 0-offset indices !!
      end
      write(fp, II)                                           #  Ibeg[NFREQ]
    end # for isize
  end # not with_cl  
  close(fp)
end # DumpSolverData




#######################################################################################

#######################################################################################

#######################################################################################

#######################################################################################


D      = DData()


# crtdust = "dustem_aSilx.dust"
# crtdust = "trust_silicate.dust"
crtdust   = "gs_aSilx.dust"

if (length(ARGS)>2)
  crtdust            =  ARGS[1]
  frequencyfilename  =  ARGS[2]
  solverfilename     =  ARGS[3]
else
  println("")
  println("Usage:   julia A2E_pre.jl   dustname  freq_file  solver_file_name")
  println("   e.g.  julia_A2E_pre.jl   gs_aSilx.dust  freq.dat solver.data.jl")
  println("")  
  exit(0)
end



# Read dust data from the files in crtdust text file
lines   = readlines(open(crtdust))
FILES   = Dict()
TRUST   = false
for l in lines
  global TRUST
  s = split(l)
  if (length(s)>0)
    if (s[1]=="trust")
      TRUST = true
      println("TRUST !!!")
    end
  end
  if (length(s)>1)  
    push!(FILES, s[1]=>s[2])
    if (s[1]=="bins")  # specify NE and the binning
      D.NE   = toi(s[2])
      D.NEPO = D.NE+1
      D.TMIN = tof(s[3]) * ones(Float64, D.NSIZE)
      D.TMAX = tof(s[4]) * ones(Float64, D.NSIZE)
    end
  end
end


# if one gave tmin and tmax in the crtdust file, set already D.TMIN and D.TMAX
if ("tmin" in keys(FILES))
  tmin = tof(FILES["tmin"])
end
if ("tmax1" in keys(FILES))
  tmax1 = tof(FILES["tmax1"])
end
if ("tmax2" in keys(FILES))
  tmax2 = tof(FILES["tmax2"])
end


@printf("Read files")
t0 = time()
D.NAME = FILES["prefix"]
if (occursin("dustem", crtdust))           # DustEM dust
  D.NE = 128  # ???
  read_DM_frequency(D, FILES["lambda"])
  read_DM_optical(D, FILES["optical"], FILES["phase_function"])
  read_DM_sizes(D, FILES["sizes"], tmin=tmin, tmax1=tmax1, tmax2=tmax2)
  read_DM_heat(D, FILES["heat"])
elseif ((TRUST)||(occursin("TRUST", crtdust)))   # Trust dust definition
  read_TRUST_optical(D, FILES["optical"])
  read_TRUST_sizes(D, FILES["sizes"])
  read_TRUST_heat(D, FILES["enthalpies"])
else                                       # GSET generic dust files
  read_GSET_optical(D, FILES["optical"])
  read_GSET_sizes(D, FILES["sizes"])
  read_GSET_heat(D, FILES["enthalpies"])
end
# @printf(" --- Read files: %.3f seconds\n", time()-t0)


if false
  freq    =  logspace(um2f(2000.0), um2f(0.1), 128)
else
  freq    =  readdlm(frequencyfilename)[:,1]
end
set_frequencies(D, freq)
D.BINS =  2500


if (false) # comparison to CRT dump.solve with the same frequency grid
  fp = open("freq.dat.jl","w")
  for f in freq
    @printf(fp, "%12.4e\n", f)
  end
end



if (false)
  # Check that Kabs() and SKabs() give reasonable values
  freq   =  um2f(30.0)
  total  =  0.0
  for isize in 1:D.NSIZE
    global total
    @printf(" --- size %3d   %.3e um     ABS %10.3e\n", isize, D.SIZE_A[isize]*1.0e4, SKabs(D, isize, freq))
    total += SKabs(D, isize, freq)
  end
  println("Kabs ", Kabs(D, freq), "   sum(SKabs) ", total)  
  # OPT[isize, ifreq, 4] ~  [ um, Kabs, Ksca, g ]
  # println(D.OPT[:,:,2])
  exit(0)
end

# The optical file identical to the one written from python --  mix is not yet tested !!!
t0 = time()
write_simple_dust([D,], "jl.dust")
@printf(" --- write_simple_dust: %.3f seconds\n", time()-t0)

t0 = time()
write_scattering_function([D,], "jl.dsc")
@printf(" --- write_scattering_function: %.3f seconds\n", time()-t0)

t0 = time()
DumpSolverData(D, solverfilename, freq, true)
@printf(" --- DumpSolverData: %.3f seconds\n", time()-t0)

"""
?? GSET --- E(T) slightly smaller than in c program ???   ratio 182.3/178.0
does not depend on whether E2T and T2E use linear or log interpolation.
It is a constant factor... as if one used 0.8% different grain sizes ???
"""


if (false)
  # Dump E[ie, isize]
  fp = open("jE.dump", "w")
  for isize in 1:D.NSIZE
    @printf(fp, "SIZE_A[%03d] = %12.4e\n", isize, D.SIZE_A[isize])
  end
  isize = 1
  for ie in 1:D.NEPO
    e = D.E[ie, isize]  # FOR SOME REASON STARTS AT HIGHER T THAN C-VERSION ???
    t = E2T(D, D.SIZE_A[isize], e)
    @printf(fp, "%12.4e %12.4e\n", t, e)
  end  
  close(fp)
end

