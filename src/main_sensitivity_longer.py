# Response of atmosphere to sea surface temperature bump using jax.jvp
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr
import argparse

import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
import jax_datetime as jdt

import jem
from jem.components import JCM, SlabOceanModel
from jem.mapping import BasicMapper
from jem.base.coupler import Coupler
import jem.utils.tree_tools as tree_tools
from jem.utils.checkpoints import (
    save_coupled_carry, load_coupled_carry,
    save_veros_carry, load_veros_carry,
)

from model_setup import build_model, get_ocean_surface_temperature, set_ocean_surface_temperature

import matplotlib as mplt
import matplotlib.pyplot as plt


print(f"jcm library is located at: {jcm.__file__}")
print(f"jem library is located at: {jem.__file__}")

# Check available devices
print(f"Available devices: {jax.devices()}")
print(f"Number of devices: {len(jax.devices())}")


def symmetric_levels(*arrays, n=11, n_std=2.0):
    """Build contour levels that are symmetric about zero, spanning
    +/- `n_std` standard deviations of the combined data."""
    values = jnp.concatenate([jnp.asarray(a).ravel() for a in arrays])
    vmax = n_std * float(jnp.std(values))
    if vmax == 0:
        vmax = 1.0
    return jnp.linspace(-vmax, vmax, n)


def save_sensitivity_data(
    data_file,
    lon,
    lat,
    sst_initial,
    sst_final,
    v_final,
    tangent_sst_initial,
    tangent_sst_final,
    tangent_v_final,
    ensemble_mean_of_sensitivity,
):
    """Save the jax.jvp sensitivity fields and the ensemble-mean sensitivity
    estimates to a netCDF file so that plotting can be done independently
    from the (expensive) simulation."""

    ensemble_sizes = np.array(sorted(ensemble_mean_of_sensitivity.keys()))
    ensemble_sst_final = np.stack(
        [np.asarray(ensemble_mean_of_sensitivity[n]["sst_final"]) for n in ensemble_sizes]
    )
    ensemble_v_final = np.stack(
        [np.asarray(ensemble_mean_of_sensitivity[n]["v_final"]) for n in ensemble_sizes]
    )
    ensemble_sensitivity_sst = np.stack(
        [np.asarray(ensemble_mean_of_sensitivity[n]["sensitivity_sst"]) for n in ensemble_sizes]
    )
    ensemble_sensitivity_v = np.stack(
        [np.asarray(ensemble_mean_of_sensitivity[n]["sensitivity_v"]) for n in ensemble_sizes]
    )

    ds = xr.Dataset(
        data_vars=dict(
            sst_initial=(("lon", "lat"), np.asarray(sst_initial)),
            sst_final=(("lon", "lat"), np.asarray(sst_final)),
            v_final=(("lon", "lat"), np.asarray(v_final)),
            tangent_sst_initial=(("lon", "lat"), np.asarray(tangent_sst_initial)),
            tangent_sst_final=(("lon", "lat"), np.asarray(tangent_sst_final)),
            tangent_v_final=(("lon", "lat"), np.asarray(tangent_v_final)),
            ensemble_sst_final=(("ensemble_size", "lon", "lat"), ensemble_sst_final),
            ensemble_v_final=(("ensemble_size", "lon", "lat"), ensemble_v_final),
            ensemble_sensitivity_sst=(("ensemble_size", "lon", "lat"), ensemble_sensitivity_sst),
            ensemble_sensitivity_v=(("ensemble_size", "lon", "lat"), ensemble_sensitivity_v),
        ),
        coords=dict(
            lon=("lon", np.asarray(lon)),
            lat=("lat", np.asarray(lat)),
            ensemble_size=("ensemble_size", ensemble_sizes),
        ),
    )

    print(f"Saving simulation output into: {data_file}")
    ds.to_netcdf(data_file)


def plot_sensitivity_comparison(column_specs, lon, lat, output_figure, suptitle):

    nrows = 3
    ncols = len(column_specs)
    fig, ax = plt.subplots(nrows, ncols, figsize=(6 * ncols, 14), squeeze=False)

    for col_idx, (col_title, row_specs) in enumerate(column_specs):
        for row_idx, (data, levels, kwargs, row_title) in enumerate(row_specs):
            _ax = ax[row_idx, col_idx]
            cf = _ax.contourf(lon, lat, jnp.asarray(data).transpose(), levels=levels, extend="both", **kwargs)
            fig.colorbar(cf, ax=_ax, orientation="vertical", shrink=0.85, pad=0.02)
            _ax.set_title(row_title if row_idx > 0 else f"{col_title}\n{row_title}")
            _ax.set_xlabel("longitude [deg]")
            _ax.set_ylabel("latitude [deg]")

    fig.suptitle(suptitle)
    print(f"Saving result figure into: {output_figure}")
    plt.savefig(output_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_results(data_file, output_figure, suptitle):
    """Load the saved sensitivity data and produce the comparison figure.

    This function only depends on data that has been written to disk by
    `save_sensitivity_data`, so the plotting step can be re-run without
    repeating the (expensive) simulation."""

    print(f"Loading simulation output from: {data_file}")
    with xr.open_dataset(data_file) as ds:
        lon = ds["lon"].values
        lat = ds["lat"].values

        sst_initial = ds["sst_initial"].values
        sst_final = ds["sst_final"].values
        v_final = ds["v_final"].values
        tangent_sst_initial = ds["tangent_sst_initial"].values
        tangent_sst_final = ds["tangent_sst_final"].values
        tangent_v_final = ds["tangent_v_final"].values

        ensemble_sizes = ds["ensemble_size"].values
        ensemble_sensitivity_sst = ds["ensemble_sensitivity_sst"].values
        ensemble_sensitivity_v = ds["ensemble_sensitivity_v"].values

    sst_levels = jnp.linspace(-2, 35, 11)
    v_levels = jnp.linspace(-1, 1, 11) * 5

    # Tangent/sensitivity fields can vary widely in magnitude across
    # experiments, so size their shading ranges from the data itself
    # (+/- 2 standard deviations), keeping the ranges symmetric about zero
    # and shared between the jvp and ensemble columns for fair comparison.
    tangent_sst_init_levels = symmetric_levels(tangent_sst_initial)
    tangent_sst_final_levels = symmetric_levels(tangent_sst_final, ensemble_sensitivity_sst)
    tangent_v_final_levels = symmetric_levels(tangent_v_final, ensemble_sensitivity_v)

    # Each column is a method; each row is a quantity (perturbation/initial
    # SST, SST response, meridional wind response). The reference column shows
    # the actual fields, while the jvp and ensemble columns show the
    # perturbation/sensitivity fields for direct comparison.
    reference_column = (
        "Reference run",
        [
            ((sst_initial - 273.15), sst_levels,         {},              "$\\mathrm{SST}_\\mathrm{init}$"),
            ((sst_final - 273.15),   sst_levels,         {},              "$\\mathrm{SST}_\\mathrm{final}$"),
            (v_final,                v_levels,           {"cmap": "bwr"}, "$\\mathrm{v}_\\mathrm{final}$"),
        ],
    )

    jvp_column = (
        "jax.jvp",
        [
            (tangent_sst_initial, tangent_sst_init_levels,  {"cmap": "bwr"}, "$\\partial \\mathrm{SST}_\\mathrm{init}$"),
            (tangent_sst_final,   tangent_sst_final_levels, {"cmap": "bwr"}, "$\\partial \\mathrm{SST}_\\mathrm{final}$"),
            (tangent_v_final,     tangent_v_final_levels,   {"cmap": "bwr"}, "$\\partial \\mathrm{v}_\\mathrm{final}$"),
        ],
    )

    ensemble_columns = [
        (
            f"direct (ens={int(number):d})",
            [
                (tangent_sst_initial,         tangent_sst_init_levels,  {"cmap": "bwr"}, "$\\partial \\mathrm{SST}_\\mathrm{init}$"),
                (ensemble_sensitivity_sst[i],  tangent_sst_final_levels, {"cmap": "bwr"}, "$\\partial \\mathrm{SST}_\\mathrm{final}$"),
                (ensemble_sensitivity_v[i],    tangent_v_final_levels,   {"cmap": "bwr"}, "$\\partial \\mathrm{v}_\\mathrm{final}$"),
            ],
        )
        for i, number in enumerate(ensemble_sizes)
    ]

    column_specs = [reference_column, jvp_column] + ensemble_columns
    plot_sensitivity_comparison(column_specs, lon, lat, output_figure, suptitle)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restart-dir", type=str, help="Restart dir", default=None)
    parser.add_argument("--total-simulation-days", type=int, help="Total time of simulation in days", default=5)
    parser.add_argument("--truncation-number", type=int, help="Truncation number", default=31)
    parser.add_argument("--test-ensemble-members", type=int, nargs="+", help="The ensemble members to be used", default=[1])
    parser.add_argument("--output-filename", type=str, help="The result in netcdf file.", default="sensitivity_data.nc")
    args = parser.parse_args()
    # Configurations



    start_datetime = jdt.to_datetime("2000-01-01")
    coupling_timestep = jdt.to_timedelta(1, "day")
    simulation_name = "sensitivity"
    output_dir = (Path("output") / simulation_name).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    one_second = jdt.to_timedelta(1, "second")
    truncation_number=31
    calendar = "365_day"
    test_ensemble_members = np.array(args.test_ensemble_members)
    ensemble_members = np.amax(test_ensemble_members)
    simulation_interval = jdt.to_timedelta(args.total_simulation_days, "day")

    output_file = output_dir / args.output_filename
    if not output_file.exists():

        # Build the coupled JCM + Veros + SlabOceanModel system. Packaged as a
        # function in `model_setup.py` so that other scripts (e.g. a jax.grad
        # sensitivity experiment) can build exactly the same model.
        model, config = build_model(
            truncation_number=truncation_number,
            start_datetime=start_datetime,
            coupling_timestep=coupling_timestep,
            calendar=calendar,
        )
        ocn_model = model.components["ocn"].raw_component

        print("Model info: ")
        tree_tools.print_tree(model.get_info(), root="Model")

        # Currently, we still call initialize() even if there is a restart directory assigned.
        # This is because there are some constants or model configuration that may actually
        # happen during initialization. Since how these numbers are pre-determined is not enforced
        # in our framework, the simple solution is to still call initialize, follow the workflow
        # of a completely new restart, then replace this initial_coupled_carry if restart files
        # are loaded.
        initial_coupled_carry = model.initialize()

        # Checkpoint
        if args.restart_dir is not None:
            restart_dir = Path(args.restart_dir)
            if not restart_dir.exists():
                raise FileNotFoundError(f"The specified restart directory {str(restart_dir):s} does not exist.")

            print(f"Use restart files in {str(restart_dir):s}")
            initial_coupled_carry = load_coupled_carry(
                restart_dir, ["atm", "ocn", "fakelnd"],
                component_loaders={"ocn": lambda path: load_veros_carry(path, ocn_model)},
            )

        trajectory_function = model.generate_trajectory_function(
            workflow=config["workflow"],
            iterations = int(simulation_interval / coupling_timestep),
        )

        @jax.jit
        def forecast(sst):
            # Work on a structural copy of the ocean state so that perturbing it
            # here cannot mutate (or leak tracers into) `initial_coupled_carry`,
            # which is captured by closure and must stay reusable across calls.
            ocn_state = jax.tree_util.tree_map(lambda x: x, initial_coupled_carry["ocn"]["state"])
            ocn_state = set_ocean_surface_temperature(ocn_state, sst)
            modified_carry = dict(
                initial_coupled_carry,
                ocn=dict(initial_coupled_carry["ocn"], state=ocn_state),
            )
            final_carry, _ = trajectory_function(modified_carry)
            return (
                get_ocean_surface_temperature(final_carry["ocn"]["state"]),
                final_carry["atm"]["derived"]["physics"]["_surface_flux"].v0,
            )

        sst_initial = get_ocean_surface_temperature(initial_coupled_carry["ocn"]["state"])

        # Put a point SST perturbation in the middle of domain
        shape2D = sst_initial.shape


        atm_model = model.components["atm"].raw_component
        lat = atm_model.coords.horizontal.latitudes * 180/jnp.pi
        lon = atm_model.coords.horizontal.longitudes * 180/jnp.pi
        llon, llat = jnp.meshgrid(lon, lat, indexing="ij")

        def gaussian(x, y, xc, yc, sigma_x, sigma_y):
            return jnp.exp( -  (x-xc)**2 / (2*sigma_x**2) - (y-yc)**2 / (2*sigma_y**2) )

        tangent_sst_initial = gaussian(llon, llat, 180.0, 0.0, 5, 8) * 1
        tangent_sst_initial /= jnp.sum(tangent_sst_initial**2)**0.5
        #tangent_sst_initial = jnp.zeros_like(sst_initial).at[shape2D[0]//2, shape2D[1]//2].set(1.0)

        # Use jax.jvp to obtain the sensitivity of surface meridional wind and SST
        print("Compute sensitivity using jax.jvp...")
        (sst_final, v_final), (tangent_sst_final, tangent_v_final) = jax.jvp(forecast, (sst_initial,), (tangent_sst_initial,))

        def report_tangent(name, tangent):
            tangent = jnp.asarray(tangent)
            print(
                f"[{name}] max|tangent| = {float(jnp.max(jnp.abs(tangent))):.6e}, "
                f"sum|tangent| = {float(jnp.sum(jnp.abs(tangent))):.6e}, "
                f"any nonzero = {bool(jnp.any(tangent != 0.0))}, "
                f"any nan = {bool(jnp.any(jnp.isnan(tangent)))}"
            )

        report_tangent("tangent_sst_final (jvp)", tangent_sst_final)
        report_tangent("tangent_v_final (jvp)", tangent_v_final)

        print("Compute sensitivity using direct method")
        epsilon = 0.01
        sst_final1, v_final1 = forecast(sst_initial)

        ensemble_collection = []
        for i in range(ensemble_members):
            print(f"Running ensemble member ({i:d}/{ensemble_members:d})")
            _epsilon = epsilon * (jax.random.normal(key=jax.random.PRNGKey(i)) + 1)

            _sst_final, _v_final = forecast(sst_initial + tangent_sst_initial * _epsilon)
            _sensitivity_sst = (_sst_final - sst_final1) / _epsilon
            _sensitivity_v = (_v_final - v_final1) / _epsilon

            ensemble_collection.append(dict(
                sst_final = sst_final1,
                v_final = v_final1,
                sensitivity_sst = _sensitivity_sst,
                sensitivity_v   = _sensitivity_v,
            ))

        ensemble_mean_of_sensitivity = {}
        for number in test_ensemble_members:
             ensemble_mean_of_sensitivity[number] = jax.tree.map(lambda *x: jnp.mean(jnp.stack(x), axis=0), *ensemble_collection[:number])


        print("Saving simulation output...")
        save_sensitivity_data(
            output_file,
            lon=lon,
            lat=lat,
            sst_initial=sst_initial,
            sst_final=sst_final,
            v_final=v_final,
            tangent_sst_initial=tangent_sst_initial,
            tangent_sst_final=tangent_sst_final,
            tangent_v_final=tangent_v_final,
            ensemble_mean_of_sensitivity=ensemble_mean_of_sensitivity,
        )


    if output_file.exists():

        output_figure = output_dir / "sensitivity_comparison.png"
        plot_sensitivity_results(
            output_file,
            output_figure,
            f"Response time: {simulation_interval / jdt.to_timedelta(1, 'day'):.1f} days",
        )

    else:
        raise Exception("Something went wrong. No output file detected.")
