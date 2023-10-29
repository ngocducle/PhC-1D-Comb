import numpy as np
import matplotlib.pyplot as plt

# Period (unit: micrometer)
a = 0.330

# Filling factor
f = 0.45

# Number of Plane wave
Nord = 31

# Grating thickness (unit: micrometer)
hgrating = 0.270

# Slab thickness (unit: micrometer)
hslab = 0.080

Ref = np.load(f'Ref_{a}_{np.round(f,2)}_{np.round(hgrating,3)}_{np.round(hslab,3)}_{Nord}.npy')

## Plot the figure
fig = plt.figure(figsize = (7,9))
plt.pcolormesh( kxs, freqs, Ref, cmap = 'bone', shading = 'gouraud', vmax = Ref.max())
plt.xlim(-0.15,0.15)
plt.ylim(0.38,0.48)
plt.xticks(np.arange(-0.10,0.15,0.1),fontsize = 14)
plt.yticks(np.arange(0.38,0.50,0.02),fontsize = 14)
plt.xlabel(r'$k_x a / (2 \pi)$', fontsize = 16)
plt.ylabel(r'$\omega a / (2\pi c)$', fontsize = 16)
plt.show()
