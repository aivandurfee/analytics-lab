import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. CONFIGURATION ---
NUM_PARTICLES = 150         
DAYS_TO_SIMULATE = 180      
START_LON = 145.0           # East of Japan
START_LAT_MIN = 30.0        
START_LAT_MAX = 40.0        

print("1. Loading Ocean Currents...")
try:
    ds_u = xr.open_dataset('2018_uvel.nc4', engine='h5netcdf')
    ds_v = xr.open_dataset('2018_vvel.nc4', engine='h5netcdf')

    # Load data (Index 0 = Day 1)
    u_data = ds_u['u'].isel(MT=0).squeeze().values
    v_data = ds_v['v'].isel(MT=0).squeeze().values
    lons = ds_u.Longitude.values
    lats = ds_u.Latitude.values
    print("   Data loaded successfully.")

except Exception as e:
    print(f"   ERROR: {e}")
    exit()

# --- 2. INITIALIZE PARTICLES ---
particles_x = np.full(NUM_PARTICLES, START_LON)
particles_y = np.linspace(START_LAT_MIN, START_LAT_MAX, NUM_PARTICLES)
print(f"2. Dropping {NUM_PARTICLES} particles in a line...")

# --- 3. PHYSICS ENGINE ---
def get_velocity_at_point(x, y, u_grid, v_grid, x_axis, y_axis):
    # Nearest Neighbor Lookup
    xi = (np.abs(x_axis[0, :] - x)).argmin()
    yi = (np.abs(y_axis[:, 0] - y)).argmin()
    
    u_val = u_grid[yi, xi]
    v_val = v_grid[yi, xi]
    
    # Handle Land (NaN) -> Stop moving
    if np.isnan(u_val): u_val = 0
    if np.isnan(v_val): v_val = 0
    
    return u_val, v_val

# Storage
history_x = []
history_y = []

# --- 4. RUN SIMULATION ---
dt = 6 * 3600  # 6-hour steps
meters_per_degree = 111000.0 

print(f"3. Simulating {DAYS_TO_SIMULATE} days...")

total_steps = DAYS_TO_SIMULATE * 4
for step in range(total_steps):
    for i in range(NUM_PARTICLES):
        u_curr, v_curr = get_velocity_at_point(
            particles_x[i], particles_y[i], u_data, v_data, lons, lats
        )
        
        dx = u_curr * dt
        dy = v_curr * dt
        
        particles_x[i] += dx / (meters_per_degree * np.cos(np.radians(particles_y[i])))
        particles_y[i] += dy / meters_per_degree

    if step % 4 == 0: # Save once per day
        history_x.append(particles_x.copy())
        history_y.append(particles_y.copy())
        
        if (step/4) % 30 == 0:
             print(f"   ... Day {int(step/4)} complete")

print("4. Generating Animation...")

# --- 5. VISUALIZATION (With Land Overlay) ---
fig, ax = plt.subplots(figsize=(12, 8))

# A. Render the Ocean Flow (Gray Arrows)
# We subsample [::15] so it doesn't look like a hairball
ax.quiver(lons[::15, ::15], lats[::15, ::15], 
          u_data[::15, ::15], v_data[::15, ::15], 
          color='lightgray', alpha=0.5, scale=40, zorder=1)

# B. Render the Land (The New Part!)
# We create a mask where data is NaN (which equals Land in HYCOM)
land_mask = np.isnan(u_data)

# Plot "Tan" color where the mask is True
# levels=[0.5, 1.5] tells it to only color values of 1 (True)
ax.contourf(lons, lats, land_mask, levels=[0.5, 1.5], colors=['#D2B48C'], zorder=2)
# Add a thin black coastline
ax.contour(lons, lats, land_mask, levels=[0.5], colors='black', linewidths=0.5, zorder=3)

# C. Render Particles
scatter = ax.scatter([], [], c='red', s=25, edgecolor='black', zorder=5)
title = ax.set_title("Day 0: The Launch")

# Formatting
ax.set_xlim(135, 235)
ax.set_ylim(20, 48)
ax.set_xlabel("Longitude (Degrees East)")
ax.set_ylabel("Latitude")
ax.set_facecolor('#E0F0FF') # Light Blue Ocean Background

def update(frame):
    scatter.set_offsets(np.c_[history_x[frame], history_y[frame]])
    title.set_text(f"Day {frame}: Tracking the Debris")
    return scatter, title

anim = FuncAnimation(fig, update, frames=len(history_x), interval=50, blit=False)

print("Done! Check the pop-up window.")
plt.show()