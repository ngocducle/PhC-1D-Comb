import S4 as S4
import numpy as np
import matplotlib.pyplot as plt
import time

# Start the simulation
start = time.time()

# Period (unit: micrometer)
a = 0.330

# Filling factor
f = 0.40                                                         

# Number of Plane wave
Nord = 15

# Air refractive index
n_water = 1.33

# Air permittivity
eps_water = n_water**2

# Refractive index of the medium
n = 3.15 + 0.0001j

# Medium permittivity
eps = n**2

# Water thickness
WaterThick = 0

# Grating thickness (unit: micrometer)
hgrating = 0.270

# Slab thickness (unit: micrometer)
hslab = 0.080

# Initialize the lattice
S = S4.New(Lattice = a, NumBasis = Nord)

# Define the materials
S.SetMaterial('Water', Epsilon = eps_water)
S.SetMaterial('Mater', Epsilon = eps)

# Set of wavenumber/momentum (unit: micrometer^{-1})
# This is k = (momentum/wavevector) / (2pi)
# => Has the same scale and unit as the frequency
# When we normalize to dimensionless unit,
# we only need to multiply by a, and do not need to
# divide by 2*pi
kx = np.linspace(0,0.6,80)    
#kx = np.concatenate( ( np.linspace(0,0.2,30), np.linspace(0.21,0.3,100), np.linspace(0.35,0.4,20) ) )
#np.save(f'kx_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',kx)
ky = [0.0] 
#np.save(f'ky_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',ky)

# Set of frequency (unit: c/micrometer)
# This is the frequency f, not the angular frequency: omega = 2*pi*f
freq = np.linspace(1.24,1.50,500)           
#np.save(f'freq_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',freq)       

# Define the structures
S.AddLayer( Name = 'Superstrate', Thickness = WaterThick, Material = 'Water' )
S.AddLayer( Name = 'Grating', Thickness = hgrating, Material = 'Mater' )
S.AddLayer( Name = 'Slab', Thickness = hslab, Material = 'Mater' )
S.AddLayer( Name = 'Substrate', Thickness = WaterThick, Material = 'Water')

# Set the air region in the Grating layer
S.SetRegionRectangle(
    Layer    = 'Grating',
    Material = 'Water',
    Center   = (0.0, 0.0),
    Angle    = 0,
    Halfwidths = (0.5 * a * f, 0.0)
    )

## Arrays of reflectivity, transmission, and absorption
Ref   = np.zeros( ( len(freq), len(kx) ) )
Trans = np.zeros( ( len(freq), len(kx) ) )
Abs   = np.zeros( ( len(freq), len(kx) ) )

## Calculate the reflectivity, transmisstion and absorption
for i in range( len(freq) ):
    # Set the incidence frequency
    f0 = freq[i]
    S.SetFrequency(f0)

    # The module of the momentum
    k  = f0 * n_water 

    for j in range( len(kx) ):
        # Incidence angle (in degrees)
        angle = np.arcsin( kx[j] / k ) * 180.0 / np.pi

        # Set the excitation
        S.SetExcitationPlanewave(
            IncidenceAngles = (angle, 0), # ( polar in [0,np.pi[, azimuthal in [0, 2*np.pi[ )
            sAmplitude = 1.0,
            pAmplitude = 0.0,
            Order = 0
            )

        # Obtain the incident, reflected and transmitted fluxes
        inc, r = S.GetPowerFlux(Layer = 'Superstrate')
        fw, _  = S.GetPowerFlux(Layer = 'Substrate')

        # Save to the Ref and Trans arrays
        Ref[i,j]   = np.abs( - r / inc )
        Trans[i,j] = np.abs( fw / inc )

#print('Ref = '+str(Ref)) 

## Array of Absorption
Abs = 1 - Ref - Trans 
Abs = np.log(Abs) 

## Arrays of data
kxs = np.zeros( (len(freq), len(kx)) )
freqs = np.zeros( (len(freq), len(kx)) )

for j in range( len(kx) ):
    kxs[:,j] = kx[j]

for i in range( len(freq) ):
    freqs[i,:] = freq[i]

## Duplicate the arrays of data
kxs   = np.concatenate( (- np.fliplr(kxs), kxs), axis = 1 )
freqs = np.concatenate( (np.fliplr(freqs), freqs), axis = 1 )
Ref   = np.concatenate( (np.fliplr(Ref), Ref), axis = 1)
Trans = np.concatenate( (np.fliplr(Trans), Trans), axis = 1)
Abs   = np.concatenate( (np.fliplr(Abs), Abs), axis = 1)

## Convert to the dimensionless units:
#  Frequency: c/a
#  Momentum: 2*np.pi / a
kxs = kxs * a
freqs = freqs * a

## Save the data to file
np.save(f'kxs_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',kxs)
np.save(f'freq_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',freq)
np.save(f'Ref_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',Ref)
np.save(f'Trans_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',Trans)
np.save(f'Abs_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',Abs)

# End the simulation
end = time.time()

## Plot the figure
fig, ax = plt.subplots(figsize = (7,9))
Refspec = plt.pcolormesh( kxs, freqs, Ref, cmap = 'bone', shading = 'gouraud', vmin = 0, vmax = 1 )
plt.xlim(-0.18,0.18)
plt.ylim(0.41,0.49) 
plt.xticks(np.arange(-0.10,0.15,0.1),fontsize = 14)
plt.yticks(np.arange(0.42,0.49,0.02),fontsize = 14)
plt.xlabel(r'$k_x a / (2 \pi)$', fontsize = 16)
plt.ylabel(r'$\omega a / (2\pi c)$', fontsize = 16) 
cbar = fig.colorbar(Refspec, ticks = [0, 0.25, 0.5, 0.75, 1.0])
cbar.ax.set_ylim(0,1)
cbar.ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'])
plt.title('Filling factor:'+str(f)+' - Ref', fontsize = 16) 
plt.savefig(f'Ref_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.png')

fig, ax = plt.subplots(figsize = (7,9))
Transpec = plt.pcolormesh( kxs, freqs, Trans, cmap = 'hot', shading = 'gouraud', vmin = 0, vmax = 1 )
plt.xlim(-0.18,0.18)
plt.ylim(0.41,0.49) 
plt.xticks(np.arange(-0.10,0.15,0.1),fontsize = 14)
plt.yticks(np.arange(0.42,0.49,0.02),fontsize = 14)
plt.xlabel(r'$k_x a / (2 \pi)$', fontsize = 16)
plt.ylabel(r'$\omega a / (2\pi c)$', fontsize = 16) 
cbar = fig.colorbar(Transpec, ticks = [0, 0.25, 0.5, 0.75, 1.0])
cbar.ax.set_ylim(0,1)
cbar.ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'])
plt.title('Filling factor:'+str(f)+' - Trans', fontsize = 16) 
plt.savefig(f'Trans_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.png')

fig, ax = plt.subplots(figsize = (7,9))
Abspec = plt.pcolormesh( kxs, freqs, Abs, cmap = 'hsv', shading = 'gouraud', vmin = Abs.min(), vmax = Abs.max() )
plt.xlim(-0.18,0.18)
plt.ylim(0.41,0.49) 
plt.xticks(np.arange(-0.10,0.15,0.1),fontsize = 14)
plt.yticks(np.arange(0.42,0.49,0.02),fontsize = 14)
plt.xlabel(r'$k_x a / (2 \pi)$', fontsize = 16)
plt.ylabel(r'$\omega a / (2\pi c)$', fontsize = 16) 
cbar = fig.colorbar(Abspec)
plt.title('Filling factor:'+str(f)+' - Abs', fontsize = 16)  
plt.savefig(f'Abs_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.png')

plt.show()

# Elapsed time (without time to plot the figur
print('Elapsed time =' + str(end - start) + ' s')


