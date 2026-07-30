import argparse
import traceback
import datetime
from pathlib import Path
import multiprocessing

import cftime

import xarray as xr

use_ipython = 'get_ipython' in globals()
print(f"use_ipython = {str(use_ipython)}")


import matplotlib as mplt
if not use_ipython:
    mplt.use("Agg")

import matplotlib.pyplot as plt
import cmocean as cmo
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import numpy as np

ocean_kinetic_energy_scale = 0.1

# Fixed by the coupled model's output convention (main_forward.py): each
# `atm-YYYYY.nc` / `ocn-YYYYY.nc` file is one simulation year of daily
# records, where YYYYY is the number of years elapsed since the start of
# the simulation.
DAYS_PER_YEAR = 365


def plot_simulation(detail):

    input_dir = detail["input_dir"]
    output_dir = detail["output_dir"]
    year = detail["year"]
    start_date = detail["start_date"]
    phase = detail["phase"]
    result = dict(detail=detail, status="UNKNOWN")

    try:

        output_figures = [
            Path(output_dir) / f"frame-{year*DAYS_PER_YEAR+i:06d}.png"
            for i in range(DAYS_PER_YEAR)
        ]

        if phase == "detect":
            result["needs_work"] = not all([output_figure.exists() for output_figure in output_figures])
            result["status"] = "detect_ok"
            return result

        # Load files
        plot_data = {
            component_name: xr.open_dataset(Path(input_dir) / f"{component_name:s}-{year:05d}.nc")
            for component_name in ["atm", "ocn"]
        }

        levels_q = np.linspace(1, 15, 71)
        levels_tskin = np.linspace(0, 25, 25+1)
        levels_sea_surface_salinity = np.linspace(34.5, 38, 51)

        for i, output_figure in enumerate(output_figures):

            try:
                frame = year * DAYS_PER_YEAR + i
                print(f"Plotting frame = {frame:d} (year={year:d}, day-of-year={i:d})")

                fig, axes = plt.subplots(
                    1, 3,
                    figsize=(24, 5),
                    subplot_kw=dict(
                        projection=ccrs.PlateCarree(),
                    ),
                    constrained_layout=True,
                )

                ax = axes.flatten()

                for _ax in ax.flatten():
                    _ax.gridlines(draw_labels=True)

                _data_q = plot_data["atm"]["specific_humidity"].isel(time=i, level=0)
                _data_tskin = plot_data["ocn"]["sea_surface_temperature"].isel(time=i) - 273.15
                _data_sea_surface_salinity = plot_data["ocn"]["sea_surface_salinity"].isel(time=i)

                coords = _data_q.coords
                sim_date = start_date + datetime.timedelta(days=frame)
                time_str = sim_date.strftime("%Y-%m-%d")
                lat = coords["lat"].to_numpy()
                lon = coords["lon"].to_numpy()

                def contourf(ax, data, levels, cmap, extend):
                    cyclic_data, cyclic_lon = add_cyclic_point(data.transpose().to_numpy(), coord=lon)
                    mappable = ax.contourf(
                        cyclic_lon, lat,
                        cyclic_data,
                        levels=levels,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap,
                        extend=extend,
                    )
                    cb = plt.colorbar(ax=ax, mappable=mappable, orientation='vertical', shrink=0.7, pad=0.1)
                    return cb

                # Plot the humidity field for the current time step
                ax_i = 0
                _ax = ax[ax_i]; ax_i += 1
                cb = contourf(_ax, _data_q, levels=levels_q, cmap=cmo.cm.rain, extend="both")
                cb.set_label("[g/kg]", fontsize=12)
                _ax.set_title("(a) Surface specific humidity")

                _ax = ax[ax_i]; ax_i += 1
                cb = contourf(_ax, _data_tskin, levels=levels_tskin, cmap=cmo.cm.thermal, extend="both")
                cb.set_label("[degC]", fontsize=12)
                _ax.set_title("(b) Sea surface temperature")

                _ax = ax[ax_i]; ax_i += 1
                cb = contourf(_ax, _data_sea_surface_salinity, levels=levels_sea_surface_salinity, cmap=plt.get_cmap('hot_r'), extend="both")
                cb.set_label("[PSU]", fontsize=12)
                _ax.set_title("(c) Sea surface salinity")

                fig.suptitle(f"[{time_str:s}]")

                print("Saving figure: ", output_figure)
                fig.savefig(output_figure, dpi=200, bbox_inches="tight")
                plt.close(fig)

            except Exception:

                traceback.print_exc()
                result["status"] = "ERROR_individual_frame"

        for ds in plot_data.values():
            ds.close()

    except Exception:

        traceback.print_exc()
        result["status"] = "ERROR"

    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plot daily atmosphere/ocean fields from main_forward.py output.",
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing atm-YYYYY.nc / ocn-YYYYY.nc files, e.g. output_T31/sweden_workshop")
    parser.add_argument("--output-dir", type=str, default="figures", help="Directory to save the output frames")
    parser.add_argument("--start-year", type=int, required=True, help="First simulation year (batch index, 0-based) to plot")
    parser.add_argument("--end-year", type=int, required=True, help="Last simulation year (batch index, 0-based, inclusive) to plot")
    parser.add_argument("--start-date", type=str, default="2000-01-01", help="Calendar date corresponding to day 0 of year 0, for labeling frames (should match main_forward.py's start_datetime)")
    parser.add_argument("--num-processes", type=int, default=multiprocessing.cpu_count(), help="Number of parallel processes to use")
    args = parser.parse_args()

    if args.end_year < args.start_year:
        raise ValueError(f"--end-year ({args.end_year:d}) must be >= --start-year ({args.start_year:d})")

    start_date = cftime.DatetimeNoLeap(*(int(x) for x in args.start_date.split("-")))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    years = np.arange(args.start_year, args.end_year + 1)

    input_args = []

    phase = 'detect'
    for year in years:
        detail = dict(
            input_dir=args.input_dir,
            output_dir=str(output_dir),
            year=int(year),
            start_date=start_date,
            phase=phase,
        )

        result = plot_simulation(detail)
        if result["needs_work"]:
            detail["phase"] = "work"
            input_args.append(detail)
        else:
            print(f"Year={year:d} does not need work.")

    with multiprocessing.Pool(processes=args.num_processes) as pool:
        pool.map(plot_simulation, input_args)

    print(f"All frames generated in '{output_dir!s}' folder.")
