# Plastic Hunter Analytics Suite

The **Intelligence Layer** of a Dynamic Ocean Remediation System for the North Pacific Gyre. This repository contains exploratory data analysis (EDA) and Lagrangian particle tracking tools that transform raw oceanographic current data into kinetic energy maps and simulated plastic drift trajectories.

---

## What It Does

### 1. Ocean Scanner (`EDA.py`)

Before cleaning the ocean, you must understand the battlefield. This script:

- **Loads** raw ocean current data (NetCDF format: u-velocity and v-velocity)
- **Computes** current speed as kinetic energy: `speed = √(u² + v²)`
- **Visualizes** two critical views:
  - **Speed heatmap** — Identifies the "Highway" (Kuroshio Extension) and the "Trap" (North Pacific Gyre)
  - **Vector field** — Confirms flow direction (clockwise gyre) and coordinate integrity (Japan to Hawaii)

This is physics-aware EDA: it validates that your data matches real ocean behavior, not just missing values.

---

### 2. Time Machine (`particle_simulation.py`)

Static maps lie. This script is a **Lagrangian particle tracking model** that shows where water—and plastic—actually goes:

- **Drops** 150 virtual particles along a vertical line east of Japan (latitudes 30°N–40°N)
- **Advects** them using 6-hour time steps over 180 days
- **Interpolates** velocities from the current grid (nearest-neighbor at each step)
- **Animates** the results with land overlay and coastlines

**Key insight:** Only particles entering at certain latitudes (e.g., 34–35°N) reach the gyre. Trash at 40°N gets caught in coastal eddies. This "selection bias" has direct policy implications for where to prioritize river filtration.

---

## Requirements

- **Python 3.8+**

Install dependencies:

```bash
pip install -r requirements.txt
```

| Package     | Purpose                          |
|------------|-----------------------------------|
| xarray     | NetCDF I/O and labeled arrays     |
| numpy      | Numerical operations              |
| matplotlib | Plots and animation               |
| h5netcdf   | NetCDF4 engine for `.nc4` files   |

---

## Data Requirements

Both scripts expect HYCOM-style ocean current data in the project directory:

- `2018_uvel.nc4` — Zonal (east–west) velocity component (m/s)
- `2018_vvel.nc4` — Meridional (north–south) velocity component (m/s)

**Data format:** NetCDF4 with 2D arrays for `Longitude`, `Latitude`, and a time dimension (e.g., `MT`). Land cells should be NaN.

These large datasets are excluded from the repo via `.gitignore`. Obtain them from HYCOM, Copernicus Marine Service, or your course/institution data source.

---

## Usage

1. Place `2018_uvel.nc4` and `2018_vvel.nc4` in the project root.
2. Run the Ocean Scanner:

   ```bash
   python EDA.py
   ```

3. Run the particle simulation:

   ```bash
   python particle_simulation.py
   ```

Both scripts open matplotlib windows. The simulation generates an animation of debris drift over 180 days.

---

## Project Context

This forms **Module 1** of a larger system. The conclusion:

> *"Static cleanup arrays fail because the trap moves. The only viable solution is a dynamic routing algorithm."*

**Module 2** will use these trajectory forecasts to optimize boat routing—"skating to where the puck is going to be"—for plastic interception.

---

## File Structure

```
Analytics Lab EDA/
├── README.md
├── requirements.txt
├── EDA.py                  # Ocean Scanner: speed map + vector field
├── particle_simulation.py  # Time Machine: Lagrangian particle tracking
├── .gitignore              # Excludes *.nc4 (large datasets)
└── 2018_uvel.nc4           # (local only, not in repo)
    2018_vvel.nc4           # (local only, not in repo)
```
