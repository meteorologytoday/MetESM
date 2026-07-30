# Response of the coupled model to an SST bump using jax.jvp, compared
# against a direct (finite-difference ensemble) estimate.
#
# The quantity whose sensitivity is studied is a "measure" -- see
# `sensitivity_measures.py`. Swapping `--measure` changes what is computed,
# saved, and plotted without touching the driver code below.
import numpy as np
import jax
import jax.numpy as jnp
import argparse
from pathlib import Path

import jcm
import jax_datetime as jdt

import jem
from jem.utils.checkpoints import load_coupled_carry, load_veros_carry
import jem.utils.tree_tools as tree_tools

from model_setup import build_model
from veros_helper import get_ocean_temperature, set_ocean_temperature
from sensitivity_measures import MEASURES, compose_measures, make_ocean_northward_heat_transport_time_mean


print(f"jcm library is located at: {jcm.__file__}")
print(f"jem library is located at: {jem.__file__}")

# Check available devices
print(f"Available devices: {jax.devices()}")
print(f"Number of devices: {len(jax.devices())}")


def report_tangent(name, tangent):
    tangent = jnp.asarray(tangent)
    print(
        f"[{name}] max|tangent| = {float(jnp.max(jnp.abs(tangent))):.6e}, "
        f"sum|tangent| = {float(jnp.sum(jnp.abs(tangent))):.6e}, "
        f"any nonzero = {bool(jnp.any(tangent != 0.0))}, "
        f"any nan = {bool(jnp.any(jnp.isnan(tangent)))}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-condition", type=str, help="The checkpoint path containing initial condition to be used", default=None)
    parser.add_argument("--total-simulation-days", type=int, help="Total time of simulation in days", default=5)
    parser.add_argument("--truncation-number", type=int, help="Truncation number", default=31)
    parser.add_argument("--jcm-timestep-min", type=int, help="JCM timestep in minutes", default=30)
    parser.add_argument("--veros-timestep-min", type=int, help="Veros timestep in minutes", default=30)
    parser.add_argument("--terrain-planet-type", type=str, help="Simulation name for output", required=True)
    parser.add_argument("--ocean-land-sea-mask-file", type=str, help="Path to the rotated-grid ocean land-sea mask file (e.g. landsea_mask_RG_4.00deg.nc)", required=True)
    parser.add_argument("--test-ensemble-members", type=int, nargs="+", help="The ensemble members to be used", default=[1])
    parser.add_argument("--output-filename", type=str, help="The result in netcdf file.", default="sensitivity_data.nc")
    parser.add_argument("--measure", type=str, nargs="+", choices=sorted(MEASURES.keys()), default=["ocean_temperature_zonal_mean"],
                         help="Which diagnostic(s) to compute the sensitivity of. Passing more than one "
                              "computes all of them from a single shared simulation.")
    parser.add_argument("--ocean-window", type=int, default=None,
                         help="Number of trailing simulated days to average over for the "
                              "ocean_northward_heat_transport_time_mean measure (default: average over "
                              "the whole trajectory). Ignored unless that measure is selected.")
    args = parser.parse_args()

    def _resolve_measure(name):
        if name == "ocean_northward_heat_transport_time_mean" and args.ocean_window is not None:
            return make_ocean_northward_heat_transport_time_mean(last_n_records=args.ocean_window)
        return MEASURES[name]

    measure = compose_measures(*[_resolve_measure(name) for name in args.measure])

    # Configurations
    start_datetime = jdt.to_datetime("2000-01-01")
    coupling_timestep = jdt.to_timedelta(1, "day")
    simulation_name = "sensitivity"
    output_dir = (Path("output") / simulation_name).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    one_second = jdt.to_timedelta(1, "second")
    truncation_number = args.truncation_number
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
            terrain_planet_type=args.terrain_planet_type,
            ocean_land_sea_mask_file=args.ocean_land_sea_mask_file,
            jcm_dt=args.jcm_timestep_min * 60.0,
            veros_dt_mom=args.veros_timestep_min * 60,
            veros_dt_tracer=args.veros_timestep_min * 60,
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
        if args.initial_condition is not None:
            initial_condition_dir = Path(args.initial_condition)
            if not initial_condition_dir.exists():
                raise FileNotFoundError(f"The specified initial condition directory {str(initial_condition_dir):s} does not exist.")

            print(f"--initial-condition is provided")
            print(f"Load initial condition: {str(initial_condition_dir):s}")
            initial_coupled_carry = load_coupled_carry(
                initial_condition_dir, ["atm", "ocn", "fakelnd"],
                component_loaders={"ocn": lambda path: load_veros_carry(path, ocn_model)},
            )

        trajectory_function = model.generate_trajectory_function(
            workflow=config["workflow"],
            iterations = int(simulation_interval / coupling_timestep),
        )

        @jax.jit
        def forecast(temp, salt, u, v):

            # Work on a structural copy of the ocean state so that perturbing it
            # here cannot mutate (or leak tracers into) `initial_coupled_carry`,
            # which is captured by closure and must stay reusable across calls.
            # Only `temp` is actually varied here -- salt/u/v are accepted for
            # a future extension but are not yet wired into the ocean state.
            ocn_state = jax.tree_util.tree_map(lambda x: x, initial_coupled_carry["ocn"]["state"])
            ocn_state = set_ocean_temperature(ocn_state, temp)

            modified_carry = dict(
                initial_coupled_carry,
                ocn=dict(
                    initial_coupled_carry["ocn"],
                    state=ocn_state,               # overwrite ocn with perturbed one
                ),
            )
            final_carry, predictions = trajectory_function(modified_carry)
            return measure.compute(final_carry, predictions)

        temp_initial = get_ocean_temperature(initial_coupled_carry["ocn"]["state"])

        atm_model = model.components["atm"].raw_component
        lat = atm_model.coords.horizontal.latitudes * 180/jnp.pi
        lon = atm_model.coords.horizontal.longitudes * 180/jnp.pi
        llon, llat = jnp.meshgrid(lon, lat, indexing="ij")

        def gaussian(x, y, xc, yc, sigma_x, sigma_y):
            return jnp.exp( -  (x-xc)**2 / (2*sigma_x**2) - (y-yc)**2 / (2*sigma_y**2) )

        # `temp_initial` spans the full water column (lon, lat, depth), so the
        # lon/lat Gaussian bump is broadcast uniformly over depth before being
        # normalized to unit L2 norm.
        tangent_temp_initial = jnp.zeros_like(temp_initial)
        
        tangent_temp_initial = tangent_temp_initial.at[:, :, -1].set(gaussian(llon, llat, 180.0, 0.0, 5, 8))
        tangent_temp_initial = tangent_temp_initial / jnp.sum(tangent_temp_initial**2)**0.5

        # Use jax.jvp to obtain the sensitivity of `measure` to the temperature perturbation.
        # salt/u/v are held fixed (None primal/tangent -- not yet used by `forecast`).
        print("Compute sensitivity using jax.jvp...")
        measure_final, tangent_measure = jax.jvp(
            forecast,
            (temp_initial, None, None, None),
            (tangent_temp_initial, None, None, None),
        )

        for varname, tangent in zip(measure.variable_specs.keys(), tangent_measure):
            report_tangent(f"tangent_{varname} (jvp)", tangent)

        print("Compute sensitivity using direct method")
        epsilon = 0.01
        temp_noise_magnitude = 0.001

        ensemble = []
        for i in range(ensemble_members):
            print(f"Running ensemble member ({i:d}/{ensemble_members:d})")
            _temp_perturbation = temp_noise_magnitude * jax.random.normal(shape=temp_initial.shape, key=jax.random.PRNGKey(i))
            _measure_final_perturbed = forecast(temp_initial + tangent_temp_initial * epsilon + _temp_perturbation, None, None, None)
            _sensitivity_measure = jax.tree.map(lambda a, b: (a - b) / epsilon, _measure_final_perturbed, measure_final)

            ensemble.append(_sensitivity_measure)

        print("Saving simulation output...")
        measure.save(
            output_file,
            tangent_temp_initial=tangent_temp_initial,
            tangent_measure=tangent_measure,
            ensemble=ensemble,
        )


    if output_file.exists():
        pass
        """
        output_figure = output_dir / "sensitivity_comparison.png"
        measure.plot(
            output_file,
            output_figure,
            f"Response time: {simulation_interval / jdt.to_timedelta(1, 'day'):.1f} days",
        )
        """
        print(f"Already done. File {str(output_file)} exists.")
    else:
        raise Exception("Something went wrong. No output file detected.")
