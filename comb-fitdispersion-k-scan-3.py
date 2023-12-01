import S4 as S4
import numpy as np
import scipy as scipy
from scipy import optimize
import matplotlib.pyplot as plt
import time 

###########################################################################
#
#
#       Fit the Absorption spectrum along a set of momenta
#       qx = kx * a / (2 * pi)
#       using the simple Lorentz function
#
#       This code studies the comb structure 
#
#               by Le Ngoc Duc 
#                   22/11/2023
#
###########################################################################

### Start the simulation
start = time.time()

### FUNCTION: simple Lorentz
def simpleLorentz(x, A, gamma, x0, y0):
    y = y0 + A * (gamma/2)**2 / ( (x-x0)**2 + (gamma/2)**2 )

    return y 

### Define the structure
# Period (unit: micrometer)
a = 0.330 
print('Period a = '+str(a))

# Filling factor
f = 0.455   
print('Filling factor f = '+str(f))

# Number of Plane waves
Nord = 9    
print('Number of Plane waves Nord = '+str(Nord))

# Optical properties of the environment
n_env = 1.00     
eps_env = n_env ** 2 

# Refractive index of the medium
n = 3.15 + 0.00000001j
eps = n ** 2 

# Superstrate thickness (unit: micrometer)
tsuperstrate = 0 
print('tsuperstrate = '+str(tsuperstrate))

# Grating thickness (unit: micrometer)
hgrating = 0.270 
print('hgrating = '+str(hgrating))

# Slab thickness (unit: micrometer)
hslab = 0.080 
print('hslab = '+str(hslab))

# Substrate thickness (unit: micrometer)
tsubstrate = 0 
print('tsubstrate = '+str(tsubstrate))

# Initialize the lattice
S = S4.New(Lattice = a, NumBasis = Nord)

# Define the materials 
S.SetMaterial('Env', Epsilon = eps_env)
S.SetMaterial('Mater', Epsilon = eps)

# Define the structures
S.AddLayer( Name = 'Superstrate', Thickness = tsuperstrate, Material = 'Env' )
S.AddLayer( Name = 'Grating', Thickness = hgrating, Material = 'Mater' )
S.AddLayer( Name = 'Slab', Thickness = hslab, Material = 'Mater' )
S.AddLayer( Name = 'Substrate', Thickness = tsubstrate, Material = 'Env' )

# Set the air region in the Grating layer
S.SetRegionRectangle(
    Layer = 'Grating',
    Material = 'Env',
    Center = (0.0, 0.0),
    Angle = 0,
    Halfwidths = (0.5 * f * a, 0.0)
)

### Set of momenta qx = kx * a / (2 * pi)
qx = np.concatenate( ( np.linspace(0.18, 0.01, 80), np.linspace(0.009,0.001,40) ) )  
#qx = np.linspace(0.075,0.001,75)      
np.save(f'qx_{a}_{np.round(f,4)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',qx)

### Set the frequency f * a (unit: a / lambda) for the 1st fit
### This is the frequency, not the angular frequency: omega = 2 * pi * f
famin = 0.45   
famax = 0.48    
Nfreq = 500 
fa    = np.linspace(famin, famax, Nfreq)
freqs = fa / a 

### Starting guess for gamma
gamma_start = 3.5e-3      

### Arrays of Amplitude A, width gamma, center x0, and base value y0 
Aval     = np.zeros( len(qx) )
gammaval = np.zeros( len(qx) )
x0val    = np.zeros( len(qx) )
y0val    = np.zeros( len(qx) )

### =======================================================================
### FUNCTION: Fit and scan
def RunSimulation(qx, freq):
    # Initialize the arrays for Reflectivity and Transmission
    Nfreq = len(freq)
    Ref   = np.zeros( Nfreq )
    Trans = np.zeros( Nfreq )
    Abs   = np.zeros( Nfreq )

    # Scan over the frequencies 
    for j in np.arange( Nfreq ):
        # Set the incident frequency
        f0 = freq[j]
        S.SetFrequency(f0)

        # The module of the momentum
        # k = 2.0 * np.pi * n_env / lambda = 2.0 * pi * f * n_env / c
        # q = k * a / ( 2.0 * np.pi ) = f * a * n_env 
        q = f0 * a * n_env 

        # The incident angle (in degrees)
        angle = np.arcsin( qx / q ) * 180.0 / np.pi 

        # Set the excitation
        S.SetExcitationPlanewave(
            IncidenceAngles = (angle, 0),
            sAmplitude = 1.0,
            pAmplitude = 0.0,
            Order = 0 
        )

        # Obtain the incident, reflected, and transmitted fluxes
        inc, r = S.GetPowerFlux( Layer = 'Superstrate' )
        fw, _  = S.GetPowerFlux( Layer = 'Substrate' ) 

        # Save to the Ref and Trans arrays
        Ref[j] = np.abs( - r / inc )
        Trans[j] = np.abs( fw / inc ) 

    # Array of Absorption
    Abs = 1 - Ref - Trans 
    return Abs 

### =======================================================================
### FUNCTION: Fit the simple Lorentz function 
def FitSimpleLorentz( qxval, freqs, gamma_guess ):  
    # Start the fit
    Abs = RunSimulation(qxval, freqs)
    X   = freqs
    Y   = Abs * 10**9 

    # Array of initial guess 
    A0 = max(Y) - min(Y)
    gamma0 = gammaguess 
    y0 = min(Y)
    index = np.argmax(Y)
    x0 = X[index] 
    InitVal = [ A0, gamma0, x0, y0 ] 

    # Array of Lower bound 
    Amin = max(Y) * 0.8
    gammamin = gamma0 * 0.001 
    x0min = x0 - 0.5 * gamma0
    y0min = 0.0
    LowerBound = [ Amin, gammamin, x0min, y0min ]

    # Array of Upper bound 
    Amax = max(Y) * 1.2 
    gammamax = gamma0 * 10 
    x0max = x0 + 0.5 * gamma0 
    y0max = 0.1 * max(Y)
    UpperBound = [ Amax, gammamax, x0max, y0max ] 

    # Tuple of bounds
    Bounds = ( LowerBound, UpperBound )

    print('InitVal:', InitVal)
    print('LowerBound:', LowerBound)
    print('UpperBound:', UpperBound)
    print('Optimize ...') 

    # Fit the Lorentz function
    popt, pcov = scipy.optimize.curve_fit( simpleLorentz, X, Y, p0 = InitVal, bounds = Bounds, maxfev = 1000000 )

    # Absorption of simple Lorentz
    Absfit = simpleLorentz( X, popt[0], popt[1], popt[2], popt[3] ) 

    # Define the new array of frequencies
    freqmin = popt[2] - 5 * popt[1]
    freqmax = popt[2] + 5 * popt[1]
    freqnew = np.linspace( freqmin, freqmax, Nfreq ) 

    return freqnew, popt, Y, Absfit 

### =======================================================================
### Scan over the momenta qx, calculate the Reflectivity, Transmission,
### and Absorption 

freqsim = freqs 

for i in np.arange( len(qx) ):
    # Take the value of qx
    qxval = qx[i]
    print('qx = '+str(qxval)) 

    # Guess for gamma
    if i == 0:
        gammaguess = gamma_start
    else:
        gammaguess = gammaval[i-1] 

    # FIT 1: 
    freq1, popt1, Y1, Absfit1 = FitSimpleLorentz( qxval, freqsim, gammaguess ) 

    # FIT 2: 
    freq2, popt2, Y2, Absfit2 = FitSimpleLorentz( qxval, freq1, gammaguess )
    
    # FIT 3:
    freq3, popt3, Y3, Absfit3 = FitSimpleLorentz( qxval, freq2, gammaguess )

    # The values of the parameters
    Aval[i]     = popt3[0]
    gammaval[i] = popt3[1]
    x0val[i]    = popt3[2]
    y0val[i]    = popt3[3]  

    # Plot the figure 
    fig, ax = plt.subplots(figsize = (7,9))
    Absspec = plt.plot( freq2 * a, Y3, '*', color = 'red' ) 
    Absfitspec1 = plt.plot( freqsim * a, Absfit1, color = 'red' ) 
    Absfitspec2 = plt.plot( freq1 * a, Absfit2, color = 'green' ) 
    Absfitspec3 = plt.plot( freq2 * a, Absfit3, color = 'blue' )
    plt.xlabel(r'$\omega a / (2 \pi c)$', fontsize = 14)
    plt.ylabel('Absorption', fontsize = 14)
    plt.title('qx = '+str(qxval))
    plt.savefig(f'ff_{np.round(f,4)}_{i}.png')
    plt.show() 

    # Assign the new range of frequencies for the next simulation 
    freqsim = freq3 

    # Save the arrays x0val and gammaval 
    np.save(f'x0val_{a}_{np.round(f,4)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',x0val)
    np.save(f'gammaval_{a}_{np.round(f,4)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',gammaval)

### ========================================================================
# Array of Q factor 
Qfactor = x0val / gammaval 
np.save(f'Qfactor_{a}_{np.round(f,4)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',Qfactor) 

# Convert the frequency to dimensionless unit
freqplot = x0val * a 

# Duplicate the arrays
qs = np.concatenate( (-np.flip(qx), qx), axis = 0)
freqplot = np.concatenate( (np.flip(freqplot), freqplot), axis = 0) 

# Plot the dispersion curve
fig, ax = plt.subplots(1,1,figsize = (8,9))
figomega = plt.plot(qs, freqplot, '*', color = 'red') 
plt.xlabel(r'$k_x a / (2 \pi)$', fontsize = 15)
plt.ylabel(r'$\omega a / (2 \pi c)$', fontsize = 15) 
plt.savefig(f'ff_{round(f,4)}_dispersion.png')
plt.show() 

# Plot the quality factor 
fig, ax = plt.subplots(1,1,figsize = (8,9))
figomega = plt.plot(np.log10(qx),np.log10(Qfactor), '*', color = 'red')
plt.xlabel(r'$log (k a / (2 \pi))$', fontsize = 15)
plt.ylabel('log(Q factor)', fontsize = 15)
plt.savefig(f'ff_{round(f,4)}_Qfactor.png') 
plt.show() 

### End the simulation
end = time.time()
print('Elapsed time = '+str(end - start)+' s') 