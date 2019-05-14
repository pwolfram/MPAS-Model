#!/usr/bin/env python

import xarray as xr
import numpy as np
import sys

ds = xr.open_dataset(sys.argv[1])

# remove particles on a processor and output new file
dsnew = ds.where(ds.currentBlock != 0, drop=True)
dsnew.to_netcdf(sys.argv[2], format='NETCDF3_CLASSIC')

assert (0 not in np.unique(dsnew.currentBlock)), \
        'Error, still have particles on processor 0!'

