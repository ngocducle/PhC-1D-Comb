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
f = 0.48

# Number of Plane waves
Nord = 11 

# Optical properties of the environment
n_env = 1.0
eps_env = n_env ** 2

# Refractive index of the medium
n = 3.15 + 0.0001j
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
    Center = (0.0, 0.0),
    Angle = 0,
    Halfwidths = (0.5 * a * f, 0.0)
)

# Set of incident angle 
Nangle = 40          
angles = np.linspace(20,0.001,Nangle) # degrees 
#angles = np.fliplr(angles) 

# Arrays of Amplitude A, width gamma, center x0, and base value y0 
Aval      = np.zeros( Nangle )
gammaval  = np.zeros( Nangle )
x0val     = np.zeros( Nangle )
y0val     = np.zeros( Nangle )       

# Set of frequencies (unit: c / micrometer)
# This is not the frequency f, not the angular frequency: omega = 2 * pi * f
freqmin = 1.36  
freqmax = 1.44
Nfreq   = 500 
freqs   = np.linspace(freqmin, freqmax, Nfreq) 

# Arrays of Reflectivity, Transmission, and Absorption
Ref   = np.zeros( Nfreq ) 
Trans = np.zeros( Nfreq )
Abs   = np.zeros( Nfreq )

# Initial values for the first simulation
A0      = 0.025 
gamma0  = 0.015 
x0      = 1.405 
y0      = 0 
Initval = [ A0, gamma0, x0, y0 ] 

# Windows for fitting 
LowerWindow = 5  
UpperWindow = 5 

# Lower bound  
Amin = A0 * 0.2   
gammamin = gamma0 * 0.5 
x0min = x0 * 0.80 
y0min = 0   
LowerBound = [ Amin, gammamin, x0min, y0min ]

# Upper bound 
Amax = A0 * 2  
gammamax = gamma0 * 2
x0max = x0 * 1.25 
y0max = 0.1   
UpperBound = [ Amax, gammamax, x0max, y0max ] 

# Tuple of bounds
Bounds = ( LowerBound, UpperBound ) 

# Scan over the angles, calculate the Reflectivity, Transmission, and Absorption
for i in np.arange(len(angles)):

    theta = angles[i]  

    print('Angle = '+str(theta)+' degrees') 

    # For each angle, scan over the frequency 
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
    Abs = 1 - Ref - Trans 

    # Fit the Lorentz function
    popt, pcov = scipy.optimize.curve_fit( simpleLorentz, freqs, Abs, p0 = Initval, bounds = Bounds )

    # Save the values to the arrays
    Aval[i]     = popt[0]
    gammaval[i] = popt[1]
    x0val[i]    = popt[2]
    y0val[i]    = popt[3]

    print(popt) 

    # Absorption of simple Lorentz
    Absfit = simpleLorentz(freqs, Aval[i], gammaval[i], x0val[i], y0val[i])

    # Plot the figure
    fig, ax = plt.subplots(figsize = (7,9))
    Absspec = plt.plot( freqs, Abs, '*', color = 'red' )
    Absfitspec = plt.plot( freqs, Absfit, color = 'blue' )
    plt.xlabel('Frequency', fontsize = 14)
    plt.ylabel('Absorption', fontsize = 14) 
    plt.title('Angle = '+str(theta)+' degrees') 
    plt.show() 

    # Redefine the array of frequencies  
    freqmin = x0val[i] - LowerWindow * gammaval[i]
    freqmax = x0val[i] + UpperWindow * gammaval[i]
    freqs   = np.linspace( freqmin, freqmax, Nfreq )    

    print('     freqmin = '+str(freqmin)+' , freqmax = '+str(freqmax)) 

    # Initial values for the next angle 
    Initval = popt  

    # Lower bound 
    Amin = Aval[i] / 5 
    gammamin = gammaval[i] / 2 
    x0min = x0val[i] * 0.80 
    y0min = y0val[i] / 10 
    LowerBound = [ Amin, gammamin, x0min, y0min ]

    # Upper bound 
    Amax = Aval[i] * 2  
    gammamax = gammaval[i] * 2
    x0max = x0val[i] * 1.25 
    y0max = y0val[i] * 1.25  
    UpperBound = [ Amax, gammamax, x0max, y0max ] 

    # Tuple of bounds
    Bounds = ( LowerBound, UpperBound )

    # Reinitialize the Reflectivity and Transmission
    Ref   = np.zeros( Nfreq ) 
    Trans = np.zeros( Nfreq ) 
    Abs   = np.zeros( Nfreq ) 

# Array of Q factor 
Qfactor = x0val / gammaval 

# Plot the dispersion curve
fig, ax = plt.subplots(1,1,figsize = (8,9))
figomega = plt.plot(angles, x0val, '*', color = 'red' ) 
plt.xlabel('Angle (degrees)', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15) 
plt.show()    

# Plot the quality factor 
fig, ax = plt.subplots(1,1,figsize = (8,9))
figomega = plt.plot(angles, Qfactor, '*', color = 'red' ) 
plt.xlabel('Angle (degrees)', fontsize = 15)
plt.ylabel('Q factor', fontsize = 15) 
plt.show() 

# Finish the simulation
end = time.time() 
print('Elapsed time = '+str(end - start)+' s') 

plt.show() 