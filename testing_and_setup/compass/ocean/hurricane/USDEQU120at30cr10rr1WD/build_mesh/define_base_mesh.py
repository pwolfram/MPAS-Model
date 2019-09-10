#!/usr/bin/env python
'''
name: define_base_mesh
authors: Steven Brus, Phillip J. Wolfram

This function specifies the resolution for a coastal refined mesh for Delaware Bay.
It contains the following resolution resgions:
  1) a QU 120km global background resolution
  2) 10km refinement region from the coast to past the shelf-break from North Carolina to New Hampshire
  3) 5km refinement region from the coast up to the shelf-break from Virginia to past Long Island, New York
  4) 2km refinement region inside Delaware Bay

'''
import numpy as np
import jigsaw_to_MPAS.coastal_tools as ct


def cellWidthVsLatLon():
    km = 1000.0

    params = ct.default_params

    print("****QU 120 background mesh and enhanced Atlantic (30km)****")
    params["mesh_type"] = "QU"
    params["dx_max_global"] = 120.0 * km
    params["region_box"] = ct.Atlantic
    params["restrict_box"] = ct.Atlantic_restrict
    params["plot_box"] = ct.Western_Atlantic
    params["dx_min_coastal"] = 30.0 * km
    params["trans_width"] = 5000.0 * km
    params["trans_start"] = 500.0 * km

    cell_width, lon, lat = ct.coastal_refined_mesh(params)

    print("****Northeast refinement (10km)***")
    params["region_box"] = ct.Delaware_Bay
    params["plot_box"] = ct.Western_Atlantic
    params["dx_min_coastal"] = 10.0 * km
    params["trans_width"] = 600.0 * km
    params["trans_start"] = 400.0 * km

    cell_width, lon, lat = ct.coastal_refined_mesh(
        params, cell_width, lon, lat)

    print("****Delaware regional refinement (2km)****")
    params["region_box"] = ct.Delaware_Region
    params["plot_box"] = ct.Delaware
    params["dx_min_coastal"] = 2.0 * km
    params["trans_width"] = 500.0 * km
    params["trans_start"] = 75.0 * km

    cell_width, lon, lat = ct.coastal_refined_mesh(
        params, cell_width, lon, lat)

    print("****Delaware Bay high-resolution (1km)****")
    params["region_box"] = ct.Delaware_Bay
    params["plot_box"] = ct.Delaware
    params["restrict_box"] = ct.Delaware_restrict
    params["dx_min_coastal"] = 1.0 * km
    params["trans_width"] = 200.0 * km
    params["trans_start"] = 17.0 * km

    cell_width, lon, lat = ct.coastal_refined_mesh(
        params, cell_width, lon, lat)

    return cell_width / 1000, lon, lat
