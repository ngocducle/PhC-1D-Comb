import S4 as S4
import numpy as np
import scipy as scipy
from scipy import optimize
import matplotlib.pyplot as plt

###########################################################################
#
#
#       Fit the Absorption spectrum at one value of the momenmtum
#       using the simple Lorentz function 
#
#       This sample code studies the comb structure 
# 
#                       by Le Ngoc Duc
#                       on 22/11/2023 
#
###########################################################################

# FUNCTION: simple Lorentz
def simpleLorentz(x,A,gamma,x0,y0):
    y = y0 + A * (gamma/2)**2 / ( (x-x0)**2 + (gamma/2)**2 )

    return y 

# Period (unit: micrometer)
a = 0.330 

# Filling factor
f = 0.49

# Number of Plane waves
Nord = 15

# Optical properties of the environment
n_env = 1.33 
eps_env = n_env ** 2 

# Refractive index of the medium
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

# Set of frequency f * a (unit: a / lambda)
# This is the frequency, not the angular frequency: omega = 2 * pi * f
famin = 0.445 
famax = 0.450  
Nfreq   = 500
fa      = np.linspace(famin, famax, Nfreq) 
freq    = fa / a 

#print('fa = ')
#print(fa)

#print('freq = ')
#print(freq) 

# Arrays of Reflectivity, Transmission, and Absorption
Ref     = np.zeros( Nfreq )
Trans   = np.zeros( Nfreq )
Abs     = np.zeros( Nfreq )

# Incident momentum qx = kx * a / (2.0 * np.pi)
qx = 0.10 

# Calculate the Reflectivity, Transmission, and Absorption
for i in range( Nfreq ):
    # Set the incidence frequency
    # The frequency in SetFrequency should always be 
    # the non-normalized one freq 
    f0 = freq[i] 
    S.SetFrequency(f0) 

    # The module of the momentum
    # k = 2.0 * np.pi * n_env / lambda = 2.0 * pi * f * n_env / c 
    # q = k * a / (2.0 * np.pi) = f * a * n_env 
    q = f0 * a * n_env  

    # The incident angle 
    angle = np.arcsin(qx/q) * 180.0 / np.pi 
    print('angle = '+str(angle))

    # Set the excitation
    S.SetExcitationPlanewave(
        IncidenceAngles = (angle, 0),
        sAmplitude = 1.0,
        pAmplitude = 0.0,
        Order = 0 
    )

    # Obtain the incident, reflected, and transmitted fluxes
    inc, r = S.GetPowerFlux(Layer = 'Superstrate')
    fw, _  = S.GetPowerFlux(Layer = 'Substrate')

    # Save to the Ref and Trans arrays 
    Ref[i]   = np.abs( - r / inc )
    Trans[i] = np.abs( fw / inc )

# Array of Absorption
Abs = 1 - Ref - Trans 

### Plot the figure of the Abs 
#fig, ax = plt.subplots(figsize = (7,9))
#Absspec = plt.plot( fa, Abs, '*', color = 'red' )
#plt.xlabel(r'$\omega a / (2 \pi c)$', fontsize = 14)
#plt.ylabel('Absorption', fontsize = 14)
#plt.show() 

### Initial values
A0     = 2.0e-5
gamma0 = 1.0e-3
x0     = 0.4475 
y0     = 0 
Initval = [ A0, gamma0, x0, y0 ]

### Fit the Lorentz function
popt, pcov = scipy.optimize.curve_fit(simpleLorentz, fa, Abs, p0 = Initval)

print(popt)

A     = popt[0]
gamma = popt[1]
x0    = popt[2]
y0    = popt[3]

### Absorption of simple Lorentz
Absfit = simpleLorentz( fa, A, gamma, x0, y0 ) 

### Plot the figure
fig, ax = plt.subplots(figsize = (7,9))
Absspec = plt.plot( fa, Abs, '*', color = 'red' )
Absfitspec = plt.plot( fa, Absfit, color = 'blue' )
plt.xlabel(r'$\omega a / (2 \pi c)$', fontsize = 14)
plt.ylabel('Absorption', fontsize = 14) 
plt.show() 