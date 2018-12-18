#!/usr/bin/env python

import numpy as np
import scipy as sp
import xarray as xr
import netCDF4

forcing = xr.open_dataset('forcing.nc')

# build xtime
xtime = ['0001-01-01_{:02d}:00:00'.format(atime) for atime in np.arange(24)]
xtime = ['{:64s}'.format(astr) for astr in xtime]

scaling = np.linspace(0,1.0,len(xtime)).T

# build fields
def make_data_array(values, scaling, forcing):
    #adjusted = np.sign(values)*np.sqrt(scaling[np.newaxis,:].T*1e3*abs(values))
    adjusted = np.repeat(100.0*values,len(xtime),axis=0)*scaling[:,np.newaxis]
    return adjusted

windSpeedU = make_data_array(forcing.windStressZonal.values, scaling, forcing)
windSpeedV = make_data_array(forcing.windStressMeridional.values, scaling, forcing)
atmosphericPressure = forcing.windStressZonal*0.0 + 101325.0

ncds = netCDF4.Dataset('atmospheric_forcing.nc', 'w', format='NETCDF3_64BIT_OFFSET')

ncds.createDimension('nCells', len(forcing.nCells))
ncds.createDimension('StrLen', 64)
ncds.createDimension('Time', None)

time = ncds.createVariable('xtime','S1', ('Time', 'StrLen'))
time[:] = netCDF4.stringtochar(np.asarray(xtime))

time = ncds.dimensions['Time'].name
ncells = ncds.dimensions['nCells'].name
ncds.createVariable('windSpeedU', np.float64,(time, ncells))
ncds.createVariable('windSpeedV', np.float64,(time, ncells))
ncds.createVariable('atmosPressure', np.float64,(time, ncells))

ncds.variables['windSpeedU'][:,:] = windSpeedU[:,:]
ncds.variables['windSpeedV'][:,:] = windSpeedV[:,:]
ncds.variables['atmosPressure'][:,:] = atmosphericPressure[:,:]

ncds.close()
