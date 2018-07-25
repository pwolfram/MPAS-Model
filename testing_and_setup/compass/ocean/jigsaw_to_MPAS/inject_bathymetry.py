#!/usr/bin/env python
# Simple script to inject bathymetry onto a mesh
# Phillip Wolfram, 01/19/2018

import matplotlib.pyplot as plt
from open_msh import readmsh
import numpy as np
from scipy import interpolate
import netCDF4 as nc4
import pprint

dtor = np.pi/180.0
rtod = 180.0/np.pi

if __name__ == "__main__":
    import sys


    # Path to bathymetry data and name of file
    data_path = "/users/sbrus/climate/bathy_data/SRTM15_plus/"
    data_file = "earth_relief_15s.nc"

    # Open NetCDF data file and read cooordintes
    nc_data = nc4.Dataset(data_path+data_file,"r")
    lon_data = nc_data.variables['lon'][:]*dtor
    lat_data = nc_data.variables['lat'][:]*dtor
    
    # Open NetCDF mesh file and read mesh points
    mesh_file = sys.argv[1]
    nc_mesh = nc4.Dataset(mesh_file,'r+')
    lon_mesh = np.mod(nc_mesh.variables['lonCell'][:] + np.pi, 2*np.pi)-np.pi
    lat_mesh = nc_mesh.variables['latCell'][:]

    # Setup interpolation boxes (for large bathymetry datasets)
    n = 100  
    xbox = np.linspace(-180,180,n)*dtor
    ybox = np.linspace(-90,90,n)*dtor
    dx = xbox[1]-xbox[0]
    dy = ybox[1]-ybox[0]
    boxes = []
    for i in range(n-1):
      for j in range(n-1):
        boxes.append(np.asarray([xbox[i],xbox[i+1],ybox[j],ybox[j+1]]))

    # Initialize bathymetry
    bathymetry = np.zeros(np.shape(lon_mesh))
    bathymetry.fill(np.nan)


    # Interpolate using the mesh and data points inside each box
    for i,box in enumerate(boxes):
      print i,"/",len(boxes)

      # Get data inside box (plus a small overlap region)
      overlap = 0.1
      lon_idx, = np.where((lon_data >= box[0]-overlap*dx) & (lon_data <= box[1]+overlap*dx))
      lat_idx, = np.where((lat_data >= box[2]-overlap*dy) & (lat_data <= box[3]+overlap*dy))
      xdata = lon_data[lon_idx]
      ydata = lat_data[lat_idx]
      zdata = nc_data.variables['z'][lat_idx,lon_idx]

      # Get mesh points inside box
      lon_idx, = np.where((lon_mesh >= box[0]) & (lon_mesh <= box[1]))
      lat_idx, = np.where((lat_mesh >= box[2]) & (lat_mesh <= box[3]))
      idx = np.intersect1d(lon_idx,lat_idx)
      xmesh = lon_mesh[idx]
      ymesh = lat_mesh[idx]
      mesh_pts = np.vstack((xmesh,ymesh)).T

      # Interpolate bathymetry onto mesh points
      bathy = interpolate.RegularGridInterpolator((xdata,ydata),zdata.T,bounds_error=False,fill_value=np.nan)
      bathy_int = bathy(mesh_pts)
      bathymetry[idx] = bathy_int

    # Create new NetCDF variables in mesh file, if necessary
    nc_vars = nc_mesh.variables.keys()
    if 'bathymetry' not in nc_vars:
      nc_mesh.createVariable('bathymetry','f8',('nCells'))
    if 'cullCell' not in nc_vars: 
      nc_mesh.createVariable('cullCell','i',('nCells'))

    # Write to mesh file
    nc_mesh.variables['bathymetry'][:] = bathymetry 
    nc_mesh.variables['cullCell'][:] = nc_mesh.variables['bathymetry'][:] > 20.0
    nc_mesh.close()


