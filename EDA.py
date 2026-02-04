import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

print("1. Loading data...")
# Load the datasets
ds_u = xr.open_dataset('2018_uvel.nc4', engine='h5netcdf')
ds_v = xr.open_dataset('2018_vvel.nc4', engine='h5netcdf')

# Extract variables (removing empty dimensions)
u = ds_u['u'].squeeze()
v = ds_v['v'].squeeze()

print("2. Calculating current speeds...")
speed = np.sqrt(u**2 + v**2)

print("3. Generating plot...")
fig, ax = plt.subplots(1, 2, figsize=(18, 8))

# --- PLOT 1: Speed Heatmap ---
speed.isel(MT=0).plot(ax=ax[0], cmap='viridis', vmin=0, vmax=1.5, cbar_kwargs={'label': 'Current Speed (m/s)'})
ax[0].set_title("Current Speed (Blue = Plastic Traps)")

# --- PLOT 2: Vector Field (The Fix is Here) ---
# FIX: We must slice [::10, ::10] on Lat/Lon too, because they are 2D arrays!
X = ds_u.Longitude[::10, ::10]
Y = ds_u.Latitude[::10, ::10]

U_sub = u.isel(MT=0)[::10, ::10]
V_sub = v.isel(MT=0)[::10, ::10]

ax[1].quiver(X, Y, U_sub, V_sub, scale=20, width=0.002)
ax[1].set_title("Surface Currents (Look for the Gyre)")
ax[1].set_xlabel("Longitude")
ax[1].set_ylabel("Latitude")

print("Done! Check the window.")
plt.show()