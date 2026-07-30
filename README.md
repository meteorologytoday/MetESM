# MetESM

This project couples the atmosphere model JAX-GCM (JCM), the ocean model Veros, and the land model JAX-LAND through JAX-ESM (JEM), in a realistic-topography setup.

## Goals

1. Set up a coupled run:
   - [x] JCM on a T31 grid
   - [x] Veros on a rotated 4-degree grid
   - [ ] JAX-LAND on a rotated 4-degree grid
2. Remapping between components:
   - [ ] Flux remapping
   - [ ] Scalar remapping
   - [ ] Vector remapping
