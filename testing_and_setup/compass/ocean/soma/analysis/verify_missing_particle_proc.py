#!/usr/bin/env python

import xarray as xr
import numpy as np
import sys

def ensure_missing_procs():
    ds = xr.open_mfdataset('particles.nc')
    procs = np.sort(np.unique(ds.currentBlock.values))
    allprocs = np.arange(0,int(sys.argv[1]))
    missingprocs = np.setdiff1d(allprocs, procs)
    if len(missingprocs) > 0:
        return True
    else:
        return False


assert ensure_missing_procs(), 'Particles are on each processor, case is not set up correctly to test error where particles are not on a processor!'
