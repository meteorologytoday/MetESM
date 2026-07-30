# Self-contained "measure" definitions for jax.jvp sensitivity experiments.
#
# A `Measure` bundles together:
#   - `compute(coupled_carry) -> jnp.ndarray`: a JAX-traceable diagnostic,
#     used inside `forecast`/`jax.jvp` in `main_sensitivity_*.py`.
#   - `dims`/`get_coords`: the named dimensions and coordinate arrays of
#     that diagnostic, for writing it to netCDF.
#   - `save`/`plot`: how to write the jax.jvp + ensemble sensitivity results
#     to netCDF and render the comparison figure.
#
# `main_sensitivity_*.py` picks one `Measure` from `MEASURES` and runs
# `forecast`/`jax.jvp`/the ensemble loop generically against
# `measure.compute`, then calls `measure.save(...)`/`measure.plot(...)`.
# Adding a new diagnostic only requires adding a new `Measure` here --
# the simulation/save/plot driver code does not change.

from dataclasses import dataclass, field
from typing import Callable, Sequence, Dict, Any

import numpy as np
import jax.numpy as jnp
import xarray as xr
import matplotlib.pyplot as plt

from jcm import constants as _jcm_constants

from veros_helper import _OCEAN_GHOST_CELL


def symmetric_levels(*arrays, n=11, n_std=2.0):
    """Build contour levels that are symmetric about zero, spanning
    +/- `n_std` standard deviations of the combined data."""
    values = jnp.concatenate([jnp.asarray(a).ravel() for a in arrays])
    vmax = n_std * float(jnp.std(values))
    if vmax == 0:
        vmax = 1.0
    return jnp.linspace(-vmax, vmax, n)


def masked_zonal_mean(field, mask, axis=0):
    """Mask-weighted mean of `field` along `axis`, skipping points where
    `mask == 0` (e.g. land points in Veros' `maskT`)."""
    mask = jnp.asarray(mask)
    weighted = jnp.sum(field * mask, axis=axis)
    count = jnp.sum(mask, axis=axis)
    return weighted / jnp.maximum(count, 1.0)



@dataclass
class Measure:
    """A single diagnostic extracted from `coupled_carry`, together with
    everything needed to save and plot its sensitivity to an SST
    perturbation."""

    variable_specs: Dict[str, Dict[str, Any]] # varname => variable information
    compute: Callable  # coupled_carry -> jnp.ndarray

    def save(
        self,
        data_file,
        *,
        tangent_temp_initial,
        tangent_measure,
        ensemble,
    ):
        """Save the jax.jvp tangent and ensemble-mean sensitivity estimates
        for this measure to a netCDF file, so that plotting can be done
        independently from the (expensive) simulation."""

        data_vars = {
            "tangent_temp_initial": (("lon", "lat", "depth"), np.asarray(tangent_temp_initial))
        }

        for i, (varname, varspec) in enumerate(self.variable_specs.items()):
            
            data_vars[f"tangent_{varname}"] = (tuple(varspec["dims"]), np.asarray(tangent_measure[i]))
            
            number_of_ensemble_members = len(ensemble)
            if number_of_ensemble_members > 0:
                
                # The i-th variable is stored in the i-th element of each ensemble member
                _ensemble_data = np.stack([np.asarray(ensemble[n][i]) for n in range(len(ensemble))])
                data_vars[f"ensemble_tangent_{varname}"] = (
                    ("ensemble",) + tuple(varspec["dims"]), _ensemble_data,
                )



        ds = xr.Dataset(data_vars=data_vars)

        print(f"Saving simulation output into: {data_file}")
        ds.to_netcdf(data_file)

def compose_measures(*measures):
    """Combine several `Measure`s into one, so a single jax.jvp/ensemble
    run can produce all of their diagnostics from one shared trajectory
    instead of rerunning the (expensive) simulation once per measure.

    Each sub-measure's `compute` is called on the same
    `(coupled_carry, predictions)`; the resulting tuples are concatenated
    in `measures` order, and `variable_specs` are merged the same way, so
    `Measure.save` writes every variable out unmodified.
    """
    variable_specs = {}
    for m in measures:
        variable_specs.update(m.variable_specs)

    def compute(coupled_carry, predictions):
        results = ()
        for m in measures:
            results += m.compute(coupled_carry, predictions)
        return results

    return Measure(variable_specs=variable_specs, compute=compute)


# ---------------------------------------------------------------------------
# Concrete measures
# ---------------------------------------------------------------------------

def _lonlat_coords(model, ocn_model):
    atm_model = model.components["atm"].raw_component
    lon = np.asarray(atm_model.coords.horizontal.longitudes) * 180 / np.pi
    lat = np.asarray(atm_model.coords.horizontal.latitudes) * 180 / np.pi
    return dict(lon=lon, lat=lat)


def _latdepth_coords(model, ocn_model):
    atm_model = model.components["atm"].raw_component
    lat = np.asarray(atm_model.coords.horizontal.latitudes) * 180 / np.pi
    depth = np.asarray(ocn_model.state.variables.zt)
    return dict(lat=lat, depth=depth)


def _ocean_temperature_zonal_mean(coupled_carry, predictions):
    g = _OCEAN_GHOST_CELL
    vs = coupled_carry["ocn"]["state"].variables
    temp = vs.temp[g:-g, g:-g, :, vs.tau]
    mask = vs.maskT[g:-g, g:-g, :]
    # Returned as a 1-tuple so the output lines up positionally with
    # `variable_specs` (and with `ensemble[n][i]` in `save`), even though
    # this measure currently has only one variable.
    return (masked_zonal_mean(temp, mask, axis=0),)

def _sea_surface_temperature(coupled_carry, predictions):
    predictionsg = _OCEAN_GHOST_CELL
    vs = coupled_carry["ocn"]["state"].variables
    temp = vs.temp[g:-g, g:-g, :, vs.tau]
    mask = vs.maskT[g:-g, g:-g, :]
    # Returned as a 1-tuple so the output lines up positionally with
    # `variable_specs` (and with `ensemble[n][i]` in `save`), even though
    # this measure currently has only one variable.
    return (masked_zonal_mean(temp, mask, axis=0),)

def _ocean_temperature(coupled_carry, predictions):
    g = _OCEAN_GHOST_CELL
    vs = coupled_carry["ocn"]["state"].variables
    temp = vs.temp[g:-g, g:-g, :, vs.tau]
    return (temp,)


OCEAN_TEMPERATURE_ZONAL_MEAN = Measure(
    variable_specs=dict(ocean_temperature_zonal_mean = dict(dims=["latitude", "depth"])),
    compute=_ocean_temperature_zonal_mean,
)

OCEAN_TEMPERATURE = Measure(
    variable_specs=dict(ocean_temperature = dict(dims=["longitude", "latitude", "depth"])),
    compute=_ocean_temperature,
)


def _ocean_zonal_velocity(coupled_carry, predictions):
    g = _OCEAN_GHOST_CELL
    vs = coupled_carry["ocn"]["state"].variables
    u = vs.u[g:-g, g:-g, :, vs.tau]
    return (u,)


def _ocean_meridional_velocity(coupled_carry, predictions):
    g = _OCEAN_GHOST_CELL
    vs = coupled_carry["ocn"]["state"].variables
    v = vs.v[g:-g, g:-g, :, vs.tau]
    return (v,)


OCEAN_ZONAL_VELOCITY = Measure(
    variable_specs=dict(ocean_zonal_velocity=dict(dims=["longitude", "latitude", "depth"])),
    compute=_ocean_zonal_velocity,
)

OCEAN_MERIDIONAL_VELOCITY = Measure(
    variable_specs=dict(ocean_meridional_velocity=dict(dims=["longitude", "latitude", "depth"])),
    compute=_ocean_meridional_velocity,
)


def _ocean_northward_heat_transport(coupled_carry, predictions):
    """Northward ocean heat transport [W] as a function of latitude,
    integrated zonally and over depth.

    Mirrors the meridional-transport pattern used by Veros' own
    `overturning` diagnostic (`dxt * cosu * v * dzt`, masked by `maskV`;
    see `veros.diagnostics.overturning.diagnose_kernel`), but multiplies by
    temperature -- averaged from consecutive T-points onto the intervening
    V-point the same way that diagnostic interpolates density onto V-faces
    -- and by `rho_0 * cp_0` to get a heat flux instead of a volume
    transport.
    """
    g = _OCEAN_GHOST_CELL
    ocn_state = coupled_carry["ocn"]["state"]
    vs = ocn_state.variables
    rho_0 = ocn_state.settings.rho_0
    cp_0 = 3991.86795711963  # J / (kg K); Veros hardcodes this per-setup, there is no settings.cp_0

    # Average temperature at consecutive T-points (j, j+1) onto the
    # intervening V-point, matching the latitude range of `v`/`maskV` ([g:-g]).
    temp_face = 0.5 * (
        vs.temp[g:-g, g:-g, :, vs.tau] + vs.temp[g:-g, g + 1:-g + 1, :, vs.tau]
    )
    v = vs.v[g:-g, g:-g, :, vs.tau]
    mask = vs.maskV[g:-g, g:-g, :]

    fac = vs.dxt[g:-g, None, None] * vs.cosu[None, g:-g, None] * vs.dzt[None, None, :]
    heat_flux = rho_0 * cp_0 * v * temp_face * mask * fac

    # Sum over longitude (axis 0) and depth (axis 2), leaving a function
    # of latitude (the ocean's V-grid) only. Returned as a 1-tuple so the
    # output lines up positionally with `variable_specs`.
    return (jnp.sum(heat_flux, axis=(0, 2)),)


OCEAN_NORTHWARD_HEAT_TRANSPORT = Measure(
    variable_specs=dict(ocean_northward_heat_transport=dict(dims=["latitude"])),
    compute=_ocean_northward_heat_transport,
)


def make_ocean_northward_heat_transport_time_mean(last_n_records=None):
    """Build a `Measure` for the time-mean northward ocean heat transport,
    averaged over the last `last_n_records` simulated days (or the whole
    trajectory, if `last_n_records` is `None`).

    Same diagnostic as `_ocean_northward_heat_transport`, but averaged over
    simulated days instead of read from the final day only.

    `predictions["ocn"]["temp"]`/`["v"]` (stacked once per day by
    `jem.components.Veros`'s step function) already have the leading time
    axis and are ghost-cell-trimmed to the same interior range as `maskV`,
    so unlike the instantaneous version above they no longer have the extra
    T-point beyond the last V-point needed to average temperature onto the
    northernmost V-face exactly. That missing neighbor is approximated by
    replicating the last available T-point (equivalent to assuming zero
    meridional temperature gradient across that one boundary face); `maskV`
    is expected to mask out that face in practice.

    `mask`/grid-metric arrays (`maskV`, `dxt`, `cosu`, `dzt`) are static, so
    they're still read once from the final `coupled_carry` state.
    """

    def _compute(coupled_carry, predictions):
        g = _OCEAN_GHOST_CELL
        ocn_state = coupled_carry["ocn"]["state"]
        vs = ocn_state.variables
        rho_0 = ocn_state.settings.rho_0
        cp_0 = 3991.86795711963  # J / (kg K); Veros hardcodes this per-setup, there is no settings.cp_0

        temp = predictions["ocn"]["temp"]  # (time, lon, lat, depth)
        v = predictions["ocn"]["v"]        # (time, lon, lat, depth)
        if last_n_records is not None:
            temp = temp[-last_n_records:]
            v = v[-last_n_records:]

        # Replicate the last latitude row to stand in for the trimmed-off
        # (j+1) neighbor, then average consecutive T-points onto the
        # intervening V-point, as in the instantaneous version.
        temp_padded = jnp.pad(temp, ((0, 0), (0, 0), (0, 1), (0, 0)), mode="edge")
        temp_face = 0.5 * (temp + temp_padded[:, :, 1:, :])

        mask = vs.maskV[g:-g, g:-g, :]
        fac = vs.dxt[g:-g, None, None] * vs.cosu[None, g:-g, None] * vs.dzt[None, None, :]
        heat_flux = rho_0 * cp_0 * v * temp_face * mask[None, ...] * fac[None, ...]

        # Sum over longitude (axis 1) and depth (axis 3), then average over
        # time (axis 0), leaving a function of latitude only.
        return (jnp.mean(jnp.sum(heat_flux, axis=(1, 3)), axis=0),)

    return Measure(
        variable_specs=dict(ocean_northward_heat_transport_time_mean=dict(dims=["latitude"])),
        compute=_compute,
    )


# Default: average over the whole trajectory. Use
# `make_ocean_northward_heat_transport_time_mean(last_n_records=...)` directly
# for a shorter trailing window.
OCEAN_NORTHWARD_HEAT_TRANSPORT_TIME_MEAN = make_ocean_northward_heat_transport_time_mean()

# Atmosphere measures
#
# `coupled_carry["atm"]["state"]` stays in spectral (modal) space, so it
# cannot be read directly the way ocean measures read `coupled_carry["ocn"]`.
# Grid-point atmosphere fields are only available through `predictions["atm"]`
# (the second argument to `compute`), whose `.dynamics` is the dycore's
# gridpoint `PhysicsState` (temperature/u_wind/v_wind, shape
# `(level, lon, lat)`) and whose `.physics` is the same physics-package
# diagnostics dict (`_convection`, `_condensation`, ...) used elsewhere in
# this repo (e.g. `jem.components.JCM`'s freshwater-flux calculation). Both
# carry a leading time axis over the whole trajectory (one entry per
# simulated day), so `[-1]` selects the end-of-trajectory field, mirroring
# how the ocean measures read `final_carry`.

def _atm_temperature(coupled_carry, predictions):
    temp = predictions["atm"].dynamics.temperature[-1]
    return (temp,)


def _atm_zonal_wind(coupled_carry, predictions):
    """Zonal wind [m/s] at each level/longitude/latitude, at the end of
    the trajectory."""
    u = predictions["atm"].dynamics.u_wind[-1]
    return (u,)


def _atm_meridional_wind(coupled_carry, predictions):
    """Meridional wind [m/s] at each level/longitude/latitude, at the end
    of the trajectory."""
    v = predictions["atm"].dynamics.v_wind[-1]
    return (v,)


def _atm_geopotential_height(coupled_carry, predictions):
    """Geopotential height [m] at each level/longitude/latitude, at the
    end of the trajectory (`dynamics.geopotential`, in J/kg, divided by
    `g` to convert to meters)."""
    z = predictions["atm"].dynamics.geopotential[-1] / _jcm_constants.grav
    return (z,)


def _atm_precipitation_zonal_mean(coupled_carry, predictions):
    """Zonal-mean total precipitation rate [kg/m^2/s, SI] as a function of
    latitude, at the end of the trajectory. Combines convective
    (`_convection.precnv`) and large-scale (`_condensation.precls`) rates,
    the same two terms `jem.components.JCM` sums into `total_freshwater_flux`.
    Both are documented as g/(m^2 s), so the sum is divided by 1000 to reach
    the SI unit.
    """
    physics = predictions["atm"].physics
    precip_g_m2_s = physics["_convection"].precnv[-1] + physics["_condensation"].precls[-1]
    precip_kg_m2_s = precip_g_m2_s / 1000.0
    return (jnp.mean(precip_kg_m2_s, axis=0),)

def _atm_precipitation(coupled_carry, predictions):
    """Zonal-mean total precipitation rate [kg/m^2/s, SI] as a function of
    latitude, at the end of the trajectory. Combines convective
    (`_convection.precnv`) and large-scale (`_condensation.precls`) rates,
    the same two terms `jem.components.JCM` sums into `total_freshwater_flux`.
    Both are documented as g/(m^2 s), so the sum is divided by 1000 to reach
    the SI unit.
    """
    physics = predictions["atm"].physics
    precip_g_m2_s = physics["_convection"].precnv[-1] + physics["_condensation"].precls[-1]
    precip_kg_m2_s = precip_g_m2_s / 1000.0
    return (precip_kg_m2_s, )



def _atm_zonal_wind_zonal_mean(coupled_carry, predictions):
    """Zonal-mean zonal wind [m/s] as a function of level and latitude,
    at the end of the trajectory."""
    u = predictions["atm"].dynamics.u_wind[-1]
    return (jnp.mean(u, axis=1),)


def _atm_meridional_wind_zonal_mean(coupled_carry, predictions):
    """Zonal-mean meridional wind [m/s] as a function of level and
    latitude, at the end of the trajectory."""
    v = predictions["atm"].dynamics.v_wind[-1]
    return (jnp.mean(v, axis=1),)


ATM_TEMPERATURE = Measure(
    variable_specs=dict(atm_temperature=dict(dims=["level", "longitude", "latitude"])),
    compute=_atm_temperature,
)

ATM_GEOPOTENTIAL_HEIGHT = Measure(
    variable_specs=dict(atm_geopotential_height=dict(dims=["level", "longitude", "latitude"])),
    compute=_atm_geopotential_height,
)

ATM_PRECIPITATION = Measure(
    variable_specs=dict(atm_precipitation=dict(dims=["longitude", "latitude"])),
    compute=_atm_precipitation,
)


ATM_PRECIPITATION_ZONAL_MEAN = Measure(
    variable_specs=dict(atm_precipitation_zonal_mean=dict(dims=["latitude"])),
    compute=_atm_precipitation_zonal_mean,
)

ATM_ZONAL_WIND_ZONAL_MEAN = Measure(
    variable_specs=dict(atm_zonal_wind_zonal_mean=dict(dims=["level", "latitude"])),
    compute=_atm_zonal_wind_zonal_mean,
)

ATM_MERIDIONAL_WIND_ZONAL_MEAN = Measure(
    variable_specs=dict(atm_meridional_wind_zonal_mean=dict(dims=["level", "latitude"])),
    compute=_atm_meridional_wind_zonal_mean,
)

ATM_ZONAL_WIND = Measure(
    variable_specs=dict(atm_zonal_wind=dict(dims=["level", "longitude", "latitude"])),
    compute=_atm_zonal_wind,
)

ATM_MERIDIONAL_WIND = Measure(
    variable_specs=dict(atm_meridional_wind=dict(dims=["level", "longitude", "latitude"])),
    compute=_atm_meridional_wind,
)


MEASURES = {
    "ocean_temperature": OCEAN_TEMPERATURE,
    "ocean_temperature_zonal_mean": OCEAN_TEMPERATURE_ZONAL_MEAN,
    "ocean_zonal_velocity": OCEAN_ZONAL_VELOCITY,
    "ocean_meridional_velocity": OCEAN_MERIDIONAL_VELOCITY,
    "ocean_northward_heat_transport": OCEAN_NORTHWARD_HEAT_TRANSPORT,
    "ocean_northward_heat_transport_time_mean": OCEAN_NORTHWARD_HEAT_TRANSPORT_TIME_MEAN,
    "atm_temperature": ATM_TEMPERATURE,
    "atm_geopotential_height": ATM_GEOPOTENTIAL_HEIGHT,
    "atm_precipitation": ATM_PRECIPITATION,
    "atm_precipitation_zonal_mean": ATM_PRECIPITATION_ZONAL_MEAN,
    "atm_zonal_wind_zonal_mean": ATM_ZONAL_WIND_ZONAL_MEAN,
    "atm_meridional_wind_zonal_mean": ATM_MERIDIONAL_WIND_ZONAL_MEAN,
    "atm_zonal_wind": ATM_ZONAL_WIND,
    "atm_meridional_wind": ATM_MERIDIONAL_WIND,
}
