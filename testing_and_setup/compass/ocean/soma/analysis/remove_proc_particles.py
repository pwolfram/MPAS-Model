#!/usr/bin/env python

import xarray as xr
import numpy as np

ds = xr.open_dataset('particles_old.nc')

# remove particles on a processor and output new file
dsnew = ds.where(ds.currentBlock != 0, drop=True)
dsnew.to_netcdf('particles.nc')

assert (0 not in np.unique(dsnew.currentBlock)), \
        'Error, still have particles on processor 0!'

