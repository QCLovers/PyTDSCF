import netCDF4 as nc
import numpy as np

def read_nc(filename: str, sites: list[tuple[int, int]]) -> dict:
    data = {}
    with nc.Dataset(filename, "r") as file:
        time_data = np.array(file.variables["time"][:])
        data["time"] = time_data
        for key in sites:
            varname = f"rho_{key}_0"
            if varname in file.variables:
                density_data_real = file.variables[varname][
                    :
                ]["real"]
                density_data_imag = file.variables[varname][
                    :
                ]["imag"]
            else:
                raise ValueError(f"Density data for site {key} {varname=} not found in {filename}")
            data[key] = np.array(density_data_real) + 1.0j * np.array(
                density_data_imag
            )

    return data
