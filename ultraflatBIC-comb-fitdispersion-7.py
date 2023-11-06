import S4 as S4
import numpy as np
import scipy as scipy
from scipy import optimize
import matplotlib.pyplot as plt
import time

# Start the simulation
start = time.time()

# FUNCTION: simple Lorentz
def simpleLorentz(x,A,gamma,x0,y0):
    y = y0 + A * (gamma/2)**2 / ( (x-x0)**2 + (gamma/2)**2 )

    return y

# Period (unit: micrometer)
a = 0.330 

# Filling factor
f = 0.51         
print('ff = '+str(f)) 

# Number of Plane waves
Nord = 11

# Optical properties of the environment
n_env = 1.0
eps_env = n_env ** 2

# Optical properties of the material
n = 3.15 + 0.00000001j  
eps = n ** 2 

# Superstrate thickness (unit: micrometer)
tsuperstrate = 0

# Grating thickness (unit: micrometer)
hgrating = 0.270

# Slab thickness (unit: micrometer)
hslab = 0.080

# Substrate thickness (unit: micrometer)
tsubstrate = 0

# Initialize the lattice
S = S4.New(Lattice = a, NumBasis = Nord)

# Define the materials
S.SetMaterial('Env', Epsilon = eps_env)
S.SetMaterial('Mater', Epsilon = eps)

# Define the structures 
S.AddLayer(Name = 'Superstrate', Thickness = tsuperstrate, Material = 'Env')
S.AddLayer(Name = 'Grating', Thickness = hgrating, Material = 'Mater')
S.AddLayer(Name = 'Slab', Thickness = hslab, Material = 'Mater')
S.AddLayer(Name = 'Substrate', Thickness = tsubstrate, Material = 'Env')

# Set the air region in the Grating layer
S.SetRegionRectangle(
    Layer = 'Grating',      
    Material = 'Env',
    Center   = (0.0, 0.0),
    Angle    = 0,
    Halfwidths = (0.5 * a * f, 0.0)
)

# Set of incident angles 
#angles = np.concatenate( (np.linspace(20,11.001,40), np.linspace(11,9.001,700), np.linspace(9,1.001,50), np.linspace(1,0.01,100) ), axis = 0 )
#angles = np.concatenate( (np.linspace(20,1.001,50), np.linspace(1,0.01,50)), axis = 0 )
angles = np.linspace(9,9.922,100)    

# Set of frequencies (unit: c / micrometer) 
# This is not the frequency f, not the angular frequency: omega = 2 * pi * f
freqmin = 1.3954 
freqmax = 1.3960          
Nfreq   = 500
freqs   = np.linspace(freqmin, freqmax, Nfreq) 

# Starting guess
gamma_start = 0.0002     

# Arrays of Amplitude A, width gamma, center x0, and base value y0 
Aval      = np.zeros( len(angles) )
gammaval  = np.zeros( len(angles) )
x0val     = np.zeros( len(angles) )
y0val     = np.zeros( len(angles) ) 

### ==============================================================================
### FUNCTION: Fit and scan
def RunSimulation(theta, freqs):
    # Initialize the arrays for Reflectivity and Transmission
    Nfreq = len(freqs)
    Ref   = np.zeros( Nfreq )
    Trans = np.zeros( Nfreq )
    Abs   = np.zeros( Nfreq ) 

    # Scan over the frequencies
    for j in np.arange(len(freqs)):
        # Set the incident frequency
        f0 = freqs[j]
        S.SetFrequency(f0)

        # The module of the momentum
        k = f0 * n_env

        # Set the excitation
        S.SetExcitationPlanewave(
            IncidenceAngles = (theta,0), # (polar in [0,np.pi[, azimuthal in [0,2*np.pi[)
            sAmplitude = 1.0,
            pAmplitude = 0.0,
            Order = 0 
        )

        # Obtain the incident, reflected, and transmitted fluxes
        inc, r = S.GetPowerFlux(Layer = 'Superstrate')
        fw, _  = S.GetPowerFlux(Layer = 'Substrate') 

        # Save to the Ref and Trans arrays
        Ref[j] = np.abs( - r / inc )
        Trans[j] = np.abs( fw / inc ) 

    # Array of Absorption
    Abs = 1 - Ref  - Trans
    return Abs 


### ==============================================================================
### Scan over the angles, calculate the Reflectivity, Transmission, and Absorption
for i in np.arange(len(angles)):

    #print(i) 
    theta = angles[i]

    print('Angle = '+str(theta)+' degrees')

    ## FIT 1:
    Abs = RunSimulation(theta, freqs) 
    #print(Abs) 
    X = freqs
    Y = Abs * 10**9  

    # Array of initial guess 
    A0 = max(Y) - min(Y)

    if i == 0:
        gamma0 = gamma_start
    else:
        gamma0 = gammaval[i-1] 

    #gamma0 = gamma_start 

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

    print('InitVal:',InitVal)
    print('LowerBound:',LowerBound)
    print('UpperBound:',UpperBound) 
    print('Optimize ...') 
 
    # Fit the Lorentz function
    popt, pcov = scipy.optimize.curve_fit( simpleLorentz, X, Y, p0 = InitVal, bounds = Bounds, maxfev = 1000000 )

    # The values of the parameters
    Aval[i]     = popt[0] 
    gammaval[i] = popt[1]
    x0val[i]    = popt[2]
    y0val[i]    = popt[3] 
    
    # Absorption of simple Lorentz
    Absfit1 = simpleLorentz(X, popt[0], popt[1], popt[2], popt[3]) 

    # Define the new array of frequencies
    freqmin = x0val[i] - 5 * gammaval[i]
    freqmax = x0val[i] + 5 * gammaval[i]
    freq1 = np.linspace(freqmin, freqmax, Nfreq) 

    ## FIT 2: 
    Abs = RunSimulation(theta, freq1) 
    #print(Abs) 
    X2 = freq1
    Y = Abs * 10**9  
    y0 = min(Y) 
    index = np.argmax(Y)
    x0 = X2[index] 

    # Array of initial guess 
    Initval = [max(Y) - min(Y), gamma0, x0, y0] 

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

    print('InitVal:',InitVal)
    print('LowerBound:',LowerBound)
    print('UpperBound:',UpperBound) 
    print('Optimize ...') 

    # Fit the Lorentz function
    popt, pcov = scipy.optimize.curve_fit( simpleLorentz, X2, Y, p0 = InitVal, bounds = Bounds, maxfev = 1000000 )

    # The values of the parameters
    Aval[i]     = popt[0] 
    gammaval[i] = popt[1]
    x0val[i]    = popt[2]
    y0val[i]    = popt[3] 
    
    # Absorption of simple Lorentz
    Absfit2 = simpleLorentz(X2, popt[0], popt[1], popt[2], popt[3]) 

    # Define the new array of frequencies
    freqmin = x0val[i] - 5 * gammaval[i]
    freqmax = x0val[i] + 5 * gammaval[i]
    freq2 = np.linspace(freqmin, freqmax, Nfreq) 

    ## FIT 3:
    Abs = RunSimulation(theta, freq2) 
    #print(Abs) 
    X3 = freq2
    Y = Abs * 10**9  
    y0 = min(Y) 
    index = np.argmax(Y)
    x0 = X3[index]

    # Array of initial guess 
    Initval = [max(Y) - min(Y), gamma0, x0, y0] 

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

    print('InitVal:',InitVal)
    print('LowerBound:',LowerBound)
    print('UpperBound:',UpperBound) 
    print('Optimize ...') 

    # Fit the Lorentz function
    popt, pcov = scipy.optimize.curve_fit( simpleLorentz, X3, Y, p0 = InitVal, bounds = Bounds, maxfev = 1000000 )

    # The values of the parameters
    Aval[i]     = popt[0] 
    gammaval[i] = popt[1]
    x0val[i]    = popt[2]
    y0val[i]    = popt[3] 

    # Absorption of simple Lorentz
    Absfit3 = simpleLorentz(X3, popt[0], popt[1], popt[2], popt[3]) 

    # Define the new array of frequencies
    freqmin = x0val[i] - 5 * gammaval[i]
    freqmax = x0val[i] + 5 * gammaval[i]
    freq3 = np.linspace(freqmin, freqmax, Nfreq) 

    ## Plot the figure
    # Plot the figure
    fig, ax = plt.subplots(figsize = (7,9))
    Absspec = plt.plot( X2, Y, '*', color = 'red' ) 
    Absfitspec1 = plt.plot( X,  Absfit1, color = 'red' ) 
    Absfitspec2 = plt.plot( X2, Absfit2, color = 'green' ) 
    Absfitspec3 = plt.plot( X3, Absfit3, color = 'blue' ) 
    plt.xlabel('Frequency', fontsize = 14)
    plt.ylabel('Absorption', fontsize = 14) 
    plt.title('Angle = '+str(theta)+' degrees') 
    plt.savefig(f'ff_{np.round(f,4)}_{i}.png')
#    plt.show()  

    freqs = freq3 

### ========================================================================
# Array of Q factor 
Qfactor = x0val / gammaval 

# Convert frequency to dimensionless units:
freqplot = x0val * a 

# Arrays of momenta 
ks = np.empty((freqplot.size))
for i in range(freqplot.size):
    ks[i] = freqplot[i] * np.sin(angles[i]*np.pi/180.0)

# Duplicate the arrays 
ks1 = np.concatenate( (-np.flip(ks), ks), axis = 0) 
freqplot = np.concatenate( (np.flip(freqplot), freqplot), axis = 0 )

# Save the arrays to files
np.save(f'angles_{a}_{np.round(f,4)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',angles)
np.save(f'ks_{a}_{np.round(f,4)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',ks)
np.save(f'ks1_{a}_{np.round(f,4)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',ks1)
np.save(f'freqplot_{a}_{np.round(f,4)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',freqplot)
np.save(f'Qfactor_{a}_{np.round(f,4)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',Qfactor)

# Plot the dispersion curve
fig, ax = plt.subplots(1,1,figsize = (8,9))
figomega = plt.plot(angles, x0val, '*', color = 'red' ) 
plt.xlabel('Angle (degrees)', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15) 
plt.savefig(f'ff_{round(f,4)}_dispersion.png')
plt.show()    

# Plot the quality factor 
fig, ax = plt.subplots(1,1,figsize = (8,9))
figomega = plt.plot(angles, np.log10(Qfactor), '*', color = 'red' ) 
plt.xlabel('Angle (degrees)', fontsize = 15)
plt.ylabel('log(Q factor)', fontsize = 15)
plt.savefig(f'ff_{round(f,4)}_Qfactor.png')  
plt.show() 

# Plot the dispersion curve
fig, ax = plt.subplots(1,1,figsize = (8,9))
figomega = plt.plot(ks1, freqplot, '*', color = 'red' ) 
plt.xlabel(r'$ka/(2\pi)$', fontsize = 15)
plt.ylabel(r'$\omega a / (2 \pi c)$', fontsize = 15) 
plt.savefig(f'ff_{round(f,4)}_dispersion-kE.png')
plt.show()   

# Plot the quality factor 
fig, ax = plt.subplots(1,1,figsize = (8,9))
figomega = plt.plot(np.log10(ks), np.log10(Qfactor), '*', color = 'red' ) 
plt.xlabel(r'$log (ka/(2\pi))$', fontsize = 15)
plt.ylabel('log(Q factor)', fontsize = 15)
plt.savefig(f'ff_{round(f,4)}_Qfactor-kE.png')  
plt.show() 

# Finish the simulation
end = time.time() 
print('Elapsed time = '+str(end - start)+' s') 

