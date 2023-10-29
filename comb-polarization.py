import S4 as S4
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import time

# Start the simulation
start = time.time()

## FUNCTION: Find the angle
def arg_angle(cos_phi, sin_phi):
    # Calculate the angle in radians given the cosinus and sinus of the angle
    #
    # Args:
    # cos_phi (float): The cosinus of the angle
    # sin_phi (float): The sinus of the angle
    #
    # Returns:
    # (float): The angle in radians, in the range [0, 2*pi)

    if sin_phi >= 0:
        phi = np.arccos(cos_phi)
    else:
        phi = 2.0 * np.pi - np.arccos(cos_phi)

    return phi

## FUNCTION: Polarization excitation
def Polarization_excitation(phi,pol):
    # Calculate the excitation amplitudes for a given polarization in S and P basis
    #
    # Args:
    # phi (float): Angle in degrees
    # pol (str): Polarization type in S and P basis. Can be one of the following:
    # 'x': Linearly polarized along the x-axis
    # 'y': Linearly polarized along the y-axis
    # 'L': Left circular polarization
    # 'R': Right circular polarization
    # 's': s-polarization
    # 'p': p-polarization
    # 'A': Antidiagonal polarization
    # 'D': Diagonal polarization
    #
    # Return:
    # tuple: A tuple containing the complex excitation amplitudes for s-polarization and p-polarization, respectively

    phi = phi * np.pi / 180

    if pol == 'x':              # x
        A_s = np.sin(phi)
        phase_s = np.pi
        A_p = np.cos(phi)
        phase_p = 0
    elif pol == 'y':            # y
        A_s = np.cos(phi)
        phase_s = 0
        A_p = np.sin(phi)
        phase_p = 0
    elif pol == 'D':            # D
        A_s = np.sin(np.pi/4 - phi)
        phase_s = 0
        A_p = np.cos(np.pi/4 - phi)
        phase_p = 0
    elif pol == 'A':            # A
        A_s = np.sin(np.pi/4 + phi)
        phase_s = np.pi
        A_p = np.cos(np.pi/4 + phi)
        phase_p = 0
    elif pol == 'S':            # S
        A_s = 1
        phase_s = 0
        A_p = 0
        phase_p = 0
    elif pol == 'P':            # P
        A_s = 0
        phase_s = 0
        A_p = 1
        phase_p = 0
    elif pol == 'L':            # L
        A_s = 1/np.sqrt(2)
        phase_s = np.pi / 2 + phi
        A_p = 1/np.sqrt(2)
        phase_p = phi
    elif pol == 'R':            # R
        A_s = 1/np.sqrt(2)
        phase_s = - np.pi/2 - phi
        A_p = 1/np.sqrt(2)
        phase_p = - phi

    sAmp = A_s * (np.cos(phase_s) + 1j*np.sin(phase_s))
    pAmp = A_p * (np.cos(phase_p) + 1j*np.sin(phase_p))

    return (sAmp,pAmp)

## FUNCTION: Run the simulation
def run_simulation(S,theta,phi,frequency,pol):
    # Calculate the normalized reflectivity and transmission for a given polarization in a
    # simulation object.
    #
    # Args:
    #   S (S^4 simulation object): The S^4 simulation object
    #   theta (float): Polar angle of the incident wave in degrees
    #   phi (float): Azimuthal angle of the incident wave in degrees
    #   frequency (float): Frequency of the excitation
    #   pol (str): Polarization type in S and P basis. Can be one of the following:
    #       'x': Linearly polarized along the x-axis
    #       'y': Linearly polarized along the y-axix
    #       'L': Left circular polarization
    #       'R': Right circular polarization
    #       's': s-polarization
    #       'p': p-polarization
    #       'A': Antidiagonal polarization
    #       'D': Diagonal polarization
    #
    # Returns:
    #   tuple: A tuple containing the normalized reflectivity and normalized transmission

    sAmp, pAmp = Polarization_excitation(phi,pol) # Polarization in S and P basis

    S.SetFrequency(frequency)

    S.SetExcitationPlanewave(
        IncidenceAngles = (theta,phi),      # In degrees
        sAmplitude = sAmp,
        pAmplitude = pAmp,
        Order = 0
        )

    inc, r = S.GetPowerFlux('AirTop')     # inc = input power
    fw, _  = S.GetPowerFlux(Layer = 'AirBottom')

    normalized_r = np.abs(r) / inc # normalized reflectivity
    normalized_t = np.abs(fw)/ inc # normalized transmission

    return (normalized_r, normalized_t)

# Period (unit: micrometer)
a = 0.330

# Filling factor
f = 0.5108

# Number of plane waves
Nord = 11  

# Initialize the lattice
S = S4.New(Lattice = a, NumBasis = Nord)

# Air refractive index
n_air = 1

# Air permittivity
eps_air = n_air ** 2

# Refractive index of the medium
n = 3.15 + 0.0001j

# Medium permittivity
eps = n ** 2

# Air thickness
AirThick = 0

# Grating thickness (unit: micrometer)
hgrating = 0.270

# Slab thickness (unit: micrometer)
hslab = 0.080

# Define the materials
S.SetMaterial('Air', Epsilon = eps_air)
S.SetMaterial('Mater', Epsilon = eps)

# Set the arrays of kx and ky (unit: micrometer^{-1})
# f = 0.5108 accidental quasi-BIC
# 0.22 <= |kx| <= 0.26
# -0.02 <= |ky| <= 0.02
kx = np.linspace(-0.32,0.32,50)
ky = np.linspace(-0.02,0.02,50)

# Set the frequency (unit: c/micrometer)
# f = 0.45 sym-BIC: freq = 0.440825 / a (Nord = 9); 0.440690 / a (Nord = 31)
# f = 0.4769 ultraflat super-BIC: freq = 0.449215 / a (Nord = 9, 11)
# f = 0.5108 sym-BIC: freq = 0.461996 / a (Nord = 9); freq = 0.461955 / a (Nord = 11)
# f = 0.5108 accidental quasi-BIC: freq = 0.460920 / a (Nord = 9); freq = 0.460810 / a (Nord = 11)
freq = 0.461996 / a

# Define the structures
S.AddLayer( Name = 'AirTop', Thickness = AirThick, Material = 'Air' )
S.AddLayer( Name = 'Grating', Thickness = hgrating, Material = 'Mater' )
S.AddLayer( Name = 'Slab', Thickness = hslab, Material = 'Mater')
S.AddLayer( Name = 'AirBottom', Thickness = AirThick, Material = 'Air')

# Set the air region in the Grating layer
S.SetRegionRectangle(
    Layer    = 'Grating',
    Material = 'Air',
    Center   = (0.0, 0.0),
    Angle    = 0,
    Halfwidths = (0.5 * a * f, 0.0)
    )

## Initialize the arrays of Reflectivity and Transmission
R_H = np.zeros((len(ky), len(kx)))
R_V = np.zeros((len(ky), len(kx)))
R_D = np.zeros((len(ky), len(kx)))
R_A = np.zeros((len(ky), len(kx)))
R_L = np.zeros((len(ky), len(kx)))
R_R = np.zeros((len(ky), len(kx)))

T_H = np.zeros((len(ky), len(kx)))
T_V = np.zeros((len(ky), len(kx)))
T_D = np.zeros((len(ky), len(kx)))
T_A = np.zeros((len(ky), len(kx)))
T_L = np.zeros((len(ky), len(kx)))
T_R = np.zeros((len(ky), len(kx)))

## We do the simulation on the plane kx-ky

for i in range( len(ky) ):

    kky = ky[i]

    for j in range( len(kx) ):

        kkx = kx[j]

        # in-plane momentum
        k = np.sqrt(  kkx**2 + kky**2 )

        # Calculate the angle phi in [0, 360 ) (degrees)
        if kkx == 0 and kky == 0:
            phi = 0
        else:
            phi = arg_angle( kkx/k, kky/k ) * 180.0 / np.pi # in degrees

        # Calculate the angle theta in [0, 180 ) (degrees)
        theta = np.arcsin( k/ freq / n_air) * 180.0 / np.pi # in degrees

        # Calculate the reflectivity and transmission for each polarization
        r_H, t_H = run_simulation( S, theta, phi, freq, 'x' )
        r_V, t_V = run_simulation( S, theta, phi, freq, 'y' )
        r_D, t_D = run_simulation( S, theta, phi, freq, 'D' )
        r_A, t_A = run_simulation( S, theta, phi, freq, 'A' )
        r_R, t_R = run_simulation( S, theta, phi, freq, 'R' )
        r_L, t_L = run_simulation( S, theta, phi, freq, 'L' )

        # Save to the arrays of reflectivity and transmission
        R_H[i,j], T_H[i,j] = r_H, t_H
        R_V[i,j], T_V[i,j] = r_V, t_V
        R_D[i,j], T_D[i,j] = r_D, t_D
        R_A[i,j], T_A[i,j] = r_A, t_A
        R_R[i,j], T_R[i,j] = r_R, t_R
        R_L[i,j], T_L[i,j] = r_L, t_L

## Arrays to store the absorption

A_H = 1 - R_H - T_H
A_V = 1 - R_V - T_V
A_D = 1 - R_D - T_D
A_A = 1 - R_A - T_A
A_R = 1 - R_R - T_R
A_L = 1 - R_L - T_L

## Compute the Stokes parameters

A10_r = A_H + A_V
A20_r = A_D + A_A
A30_r = A_L + A_R
A1_r  = A_H - A_V
A2_r  = A_D - A_A
A3_r  = A_L - A_R

s1_r  = A1_r / A10_r
s2_r  = A2_r / A20_r
s3_r  = A3_r / A30_r

rho_r = np.sqrt( s1_r**2 + s2_r**2 + s3_r**2 )

s3_rho_r = s3_r / rho_r

# Ellipticity
chi_r = np.arcsin( s3_rho_r ) / 2.0

# Orientation
cos_2phi_r = s1_r / rho_r / np.cos( 2 * chi_r )
sin_2phi_r = s2_r / rho_r / np.cos( 2 * chi_r )

phi_r = np.zeros( (kx.size, ky.size) )

for i in range( ky.size ):
    for j in range( kx.size ):
        if sin_2phi_r[i,j] >= 0: # 2*phi is in the interval [0, np.pi]
            phi_r[i,j] = 0.5 * np.arccos( cos_2phi_r[i,j] )
        elif sin_2phi_r[i,j] < 0: # 2*phi is in the interval ]-np.pi, 0]
            phi_r[i,j] = - 0.5 * np.arccos( cos_2phi_r[i,j] )

# Convert the angles from radians to degrees
chi_r = chi_r * 180.0 / np.pi
phi_r = phi_r * 180.0 / np.pi

## Arrays of data to plot the figures
kxs = np.zeros( (ky.size, kx.size) )
kys = np.zeros( (ky.size, kx.size) )

for i in range( ky.size ):
    kys[i,:] = ky[i]

for j in range( kx.size ):
    kxs[:,j] = kx[j]

## Convert to the dimensionless units:
#  Frequency: c/a
#  Momentum: 2*np.pi / a
kxs = kxs * a
kys = kys * a


#print(chi_r)
#print(phi_r)

## Save the data to arrays
np.save(f'AH_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',A_H)
np.save(f'AV_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',A_V)
np.save(f'AD_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',A_D)
np.save(f'AA_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',A_A)
np.save(f'AL_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',A_L)
np.save(f'AR_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',A_R)
np.save(f'chi_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',chi_r)
np.save(f'phi_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy',phi_r)

# End the simulation
end = time.time()
print('Elapsed time =' + str(end - start)+' s')

## Plot the figures of chi and psi
fig, ax = plt.subplots()
fig_chi = ax.pcolormesh(kxs, kys, chi_r, cmap = 'bwr', vmin = -45, vmax = 45)
#ax.set_xlim(-0.12,0.12)
#ax.set_ylim(-0.02,0.02)
ax.set_xlabel(r'$ k_x a / (2 \pi) $', fontsize = 14)
ax.set_ylabel(r'$ k_y a / (2 \pi) $', fontsize = 14)
ax.set_title(r'$ \chi $', fontsize = 14)
#ax.set_aspect('equal')
cbar = fig.colorbar(fig_chi, ticks = [-45, 0, 45])
cbar.ax.set_ylim(-45,45)
cbar.ax.set_yticklabels(['-45 (R)', '0', '45 (L)'])
plt.show()

cmap = LinearSegmentedColormap.from_list("", [(0,"#E5E5E5"),(.05,"#FFB494"),(.2,"#FF6929"),(.5,"#000000"),(.8,"#78AB30"),(.95,"#BBD598"),(1,"#E5E5E5")])

fig, ax = plt.subplots()
fig_psi = ax.pcolormesh(kxs, kys, phi_r, cmap = cmap, vmin = -90, vmax = 90)
#ax.set_xlim(-0.12,0.12)
#ax.set_ylim(-0.02,0.02)
ax.set_xlabel(r'$ k_x a / (2 \pi) $', fontsize = 14)
ax.set_ylabel(r'$ k_y a / (2 \pi) $', fontsize = 14)
ax.set_title(r'$ \phi $', fontsize = 14)
#ax.set_aspect('equal')
cbar = fig.colorbar(fig_psi, ticks = [-90, -45, 0, 45, 90])
cbar.ax.set_ylim(-90,90)
cbar.ax.set_yticklabels(['-90 (V)', '-45 (A)', '0 (H)', '45 (D)', '90 (V)'])
plt.show()
