import S4 as S4
import numpy as np
import scipy as scipy
from scipy import optimize 
import matplotlib.pyplot as plt 

# FUNCTION: simple Lorentz 
def simpleLorentz(x,A,gamma,x0,y0):
    y = y0 + A * (gamma/2)**2 / ( (x-x0)**2 + (gamma/2)**2 )

    return y  

# Check the simple Lorentz function
#x0 = 0.5
#y0 = 0 
#gamma = 0.2 
#A = 1.0  
#Nx = 500 
#x = np.linspace(x0-3*gamma,x0+3*gamma,Nx)
#y = simpleLorentz(x,A,gamma,x0,y0) 

#plt.figure(figsize=(8,8))
#plt.plot(x,y)
#plt.show() 

# Period (unit: micrometer)
a = 0.330 

# Filling factor
f = 0.48 

# Number of Plane wave
Nord = 11 

# Optical properties of the environment 
n_env = 1.0 
eps_env = n_env ** 2 

# Refractive index of the medium
n = 3.15 + 0.0001j 
eps = n ** 2

# Superstrate thickness (unit: micrometer)
tsuperstrate = 0 

# Grating thickness (unit: mirometer) 
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

# Set of wavevector/momentum (unit: micrometer^{-1})
# This is k = (momentum / wavevector) / (2pi)
# => Has the same scale and unit as the frequency 
# When we normalize to dimensionless unit,
# we only need to multiply by a, and do not need to 
# divide by 2 * pi
kx = 0 
ky = 0

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
    Halfwidths = (0.5 * a * f, 0.0)
)

# Set of frequency (unit: c / micrometer)
# This is the frequency f, not the angular frequency: omega = 2 * pi * f
freqmin = 1.36   
freqmax = 1.44        
Nfreq = 500  
freq = np.linspace(freqmin,freqmax,Nfreq)

# Arrays of reflectivity, transmission, and absorption
Ref   = np.zeros( Nfreq )
Trans = np.zeros( Nfreq )
Abs   = np.zeros( Nfreq )

#print(Ref) 

# Incident angle
angle = 20 # degrees  

# Calculate the reflectivity, transmission, and absorption
for i in range( len(freq) ):
    # Set the incidence frequency
    f0 = freq[i]
    S.SetFrequency(f0)

    # The module of the momentum
    k = f0 * n_env 

    # Set the excitation 
    S.SetExcitationPlanewave(
        IncidenceAngles = (angle, 0), # (polar in [0,np.pi[, azimuthal in [0,2*np.pi[)
        sAmplitude = 1.0,
        pAmplitude = 0.0,
        Order = 0 
    )

    # Obtain the incident, reflected and transmitted fluxes
    inc, r = S.GetPowerFlux(Layer = 'Superstrate')
    fw, _  = S.GetPowerFlux(Layer = 'Substrate')

    # Save to the Ref and Trans arrays
    Ref[i]   = np.abs( - r / inc )
    Trans[i] = np.abs( fw / inc ) 

# Array of Absorption
Abs = 1 - Ref - Trans 

# Initial values 
A0      = 0.025 
gamma0  = 0.015 
x0      = 1.405 
y0      = 0 
Initval = [ A0, gamma0, x0, y0 ]

# Fit the Lorentz function 
popt, pcov = scipy.optimize.curve_fit(simpleLorentz, freq, Abs, p0 = Initval) 

print(popt) 

A     = popt[0]
gamma = popt[1]
x0 = popt[2]
y0 = popt[3]

# Absorption of simple Lorentz 
Absfit = simpleLorentz(freq,A,gamma,x0,y0)

# Plot the figure
fig, ax = plt.subplots(figsize = (7,9))
Absspec = plt.plot( freq, Abs, '*', color = 'red' ) 
Absfitspec = plt.plot( freq, Absfit, color = 'blue' ) 
plt.xlabel('Frequency', fontsize = 14)
plt.ylabel('Absorption', fontsize = 14) 
plt.show() 


