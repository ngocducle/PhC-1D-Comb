import numpy as np
import scipy as scipy 
from scipy import interpolate 
import S4 as S4
import matplotlib.pyplot as plt
import time

# Start the simulation
start = time.time() 

# Period (unit: micrometer)
a = 0.550 
print('Period a = '+str(a))

# Thickness of the HSQ layer (unit: micrometer)
tHSQ = 0.300
print('tHSQ = '+str(tHSQ))

# Thickness of the etched a-Si layer (unit: micrometer)
t1 = 0.486 
print('t1 = '+str(t1))  

# Thickness of the residual a-Si layer (unit: micrometer)
tres = 0.00486
print('tres = '+str(tres))

# Thickness of the SiO2 layer (unit: micrometer)
tSiO2u = 0.020 
print('tSiO2 = '+str(tSiO2u))

# Thickness of the a-Si slab (unit: micrometer)
t2 = 0.1467
print('t2 = '+str(t2))

# Thickness of the SiO2 layer (unit: micrometer)
tSiO2d = 2.0
print('tSiO2 = '+str(tSiO2d)) 

# Filling factor 
f = 0.49  
print('Filling factor = '+str(f)) 

# Number of plane waves
Nord = 11 

# Array of incident angles (in degrees)
Nangle = 50 
angle  = np.linspace(0,30,Nangle)

# Optical properties of air
n_air = 1 
eps_air = n_air ** 2 

# Optical properties of water 
n_water = 1.33 
eps_water = n_water ** 2 

# Optical properties of Si substrate
n_Si = 3.48
eps_Si = n_Si ** 2

# Optical properties of SiO2 
n_SiO2 = 1.46 
eps_SiO2 = n_SiO2 ** 2 

# Optical properties of the material
print('Load the material ...')
data = np.loadtxt('aSi_He_no_Ar_25W_2T_90sccm_lowstress.txt')
wvlg_data = data[:,0]
n_data = data[:,1]
k_data = data[:,2]
f_n = interpolate.interp1d(wvlg_data, n_data)
f_k = interpolate.interp1d(wvlg_data, k_data) 
#print(f_n) 

# The range of wavelength (micrometer)
Nwvlg = 500   
wavelength = np.linspace(1.2, 1.6, Nwvlg)  
naSi = f_n(wavelength)
kaSi = f_k(wavelength)        

#print(wvlg) 
#print(n_fit) 

eps_aSi_Re = naSi**2 - kaSi**2
eps_aSi_Im = 2.0 * naSi * kaSi 
eps_aSi    = eps_aSi_Re + 1j * eps_aSi_Im 

## Plot the figure of refractive index
#fig = plt.figure(figsize = (7,9)) 
#plt.plot(wvlg_data, n_data, color='blue') 
#plt.plot(wvlg_data,k_data,'o',color='red')
#plt.plot(wvlg, naSi, linewidth=3, color='red') 
#plt.plot(wvlg,kaSi,color='green') 
#plt.show() 

### Arrays of data 
Ref   = np.zeros( (len(wavelength), len(angle)) )
Trans = np.zeros( (len(wavelength), len(angle)) )

### Scan over the wavelength
for i in range( len(wavelength) ):
    # Initialize the S4 simulation
    S = S4.New(Lattice = a, NumBasis = Nord)

    # Define the materials 
    S.SetMaterial('Air', Epsilon = eps_air)
    S.SetMaterial('Water', Epsilon = eps_water)
    S.SetMaterial('Si', Epsilon = eps_Si)
    S.SetMaterial('SiO2', Epsilon = eps_SiO2) 

    # Set the incident frequency
    wvlg = wavelength[i]
    f0 = 1 / wvlg
    S.SetFrequency(f0) 

    # The module of the momentum
    k = f0 * n_air

    # Define the material aSi
    S.SetMaterial('aSi', Epsilon = eps_aSi[i]) 

    # Define the structure
    S.AddLayer( Name = 'Superstrate', Thickness = 0, Material = 'Air' )
    S.AddLayer( Name = 'HSQ', Thickness = tHSQ, Material = 'aSi' )
    S.AddLayer( Name = 'Grating', Thickness = t1 - tres, Material = 'aSi' )
    S.AddLayer( Name = 'Residual', Thickness = tres, Material = 'aSi' ) 
    S.AddLayer( Name = 'SiO2u', Thickness = tSiO2u, Material = 'SiO2' )
    S.AddLayer( Name = 'Slab', Thickness = t2, Material = 'aSi' )
    S.AddLayer( Name = 'SiO2d', Thickness = tSiO2d, Material = 'SiO2' )
    S.AddLayer( Name = 'Substrate', Thickness = 0, Material = 'Si' ) 

    # Set the air region in the grating layer
    S.SetRegionRectangle(
        Layer    = 'HSQ',
        Material = 'Air',
        Center   = (0.0, 0.0),
        Angle    = 0,
        Halfwidths = (0.5 * a * f, 0.0) 
    )

    S.SetRegionRectangle(
        Layer    = 'Grating',
        Material = 'Air',
        Center   = (0.0, 0.0),
        Angle    = 0,
        Halfwidths = (0.5 * a * f, 0.0) 
    )

    # Scan over the angle 
    for j in range(len(angle)):
        # Incident angle (in degrees)
        theta = angle[j]

        # Set the excitation
        S.SetExcitationPlanewave(
            IncidenceAngles = (theta,0),
            sAmplitude = 1.0,
            pAmplitude = 0.0,
            Order = 0 
        )

        # Obtain the incident, reflected, and transmitted fluxes
        inc, r = S.GetPowerFlux(Layer = 'Superstrate')
        fw, _  = S.GetPowerFlux(Layer = 'Substrate')

        # Save to the Ref and Trans arrays
        Ref[i,j] = np.abs( - r / inc )
        Trans[i,j] = np.abs( fw / inc ) 

## Array of Absorption
Abs = 1 - Ref - Trans 
Abs = np.log(Abs)

# Arrays of data to plot
angles = np.zeros( (len(wavelength), len(angle)) )
for j in range(len(angle)):
    angles[:,j] = angle[j]

wavelengths = np.zeros( (len(wavelength), len(angle)) )
for i in range(len(wavelength)):
    wavelengths[i,:] = wavelength[i]

angles = np.concatenate( (-np.fliplr(angles), angles), axis = 1 )
wavelengths = np.concatenate( (np.fliplr(wavelengths), wavelengths), axis = 1 )
Refs   = np.concatenate( (np.fliplr(Ref), Ref), axis = 1 )
Transs = np.concatenate( (np.fliplr(Trans), Trans), axis = 1 )
Abss   = np.concatenate( (np.fliplr(Abs), Abs), axis = 1 )

# Finish the simulation
end = time.time()
print('Elapsed time = '+str(end - start)+' s') 

# Plot the figure
fig, ax = plt.subplots(1,1,figsize = (7,9))
Spec = plt.pcolormesh( angles, 1000*np.flipud(wavelengths), np.flipud(Refs), cmap = 'bone', shading = 'gouraud', vmin = 0, vmax = 1 )
plt.xlim(-30,30)
plt.ylim(1500,1200)
plt.xticks(np.arange(-30,40,10), fontsize = 14)
plt.yticks(np.arange(1200,1550,50),fontsize = 14)
plt.xlabel('Angle (degrees)', fontsize = 14)
plt.ylabel('Wavelength (nm)', fontsize = 14)
cbar = fig.colorbar(Spec, ticks = [0, 0.25, 0.5, 0.75, 1.0])
cbar.ax.set_yticklabels(['0','0.25','0.5','0.75','1'],fontsize = 14)
plt.title('ff = '+str(f), fontsize =14) 
plt.savefig(f'Ref_ff{np.round(f,2)}.png')
plt.show()

fig, ax = plt.subplots(1,1,figsize = (7,9))
Spec = plt.pcolormesh( angles, 1000*np.flipud(wavelengths), np.flipud(Transs), cmap = 'hot', shading = 'gouraud', vmin = 0, vmax = 1 )
plt.xlim(-30,30)
plt.ylim(1500,1200)
plt.xticks(np.arange(-30,40,10), fontsize = 14)
plt.yticks(np.arange(1200,1550,50),fontsize = 14)
plt.xlabel('Angle (degrees)', fontsize = 14)
plt.ylabel('Wavelength (nm)', fontsize = 14)
cbar = fig.colorbar(Spec, ticks = [0, 0.25, 0.5, 0.75, 1.0])
cbar.ax.set_yticklabels(['0','0.25','0.5','0.75','1'],fontsize = 14)
plt.title('ff = '+str(f), fontsize =14) 
plt.savefig(f'Trans_ff{np.round(f,2)}.png')
plt.show()

fig, ax = plt.subplots(1,1,figsize = (7,9))
Spec = plt.pcolormesh( angles, 1000*np.flipud(wavelengths), np.flipud(Abss), cmap = 'hsv', shading = 'gouraud', vmin = Abs.min(), vmax = Abs.max() )
plt.xlim(-30,30)
plt.ylim(1500,1200)
plt.xticks(np.arange(-30,40,10), fontsize = 14)
plt.yticks(np.arange(1200,1550,50),fontsize = 14)
plt.xlabel('Angle (degrees)', fontsize = 14)
plt.ylabel('Wavelength (nm)', fontsize = 14)
cbar = fig.colorbar(Spec)
#cbar.ax.set_yticklabels(['0','0.25','0.5','0.75','1'],fontsize = 14)
plt.title('ff = '+str(f), fontsize =14) 
plt.savefig(f'Abs_ff{np.round(f,2)}.png') 
plt.show()