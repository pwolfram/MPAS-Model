#!/usr/bin/env python
"""
File writes a particle input dataset for use in MPAS-O / E3SM.

Phillip J. Wolfram
Last Modified: 07/05/2018
"""

import netCDF4
import numpy as np

verticaltreatments = {'indexLevel':1, 'fixedZLevel': 2, 'passiveFloat': 3, 'buoyancySurface': 4, 'argoFloat': 5}
defaults = {'dt': 300, 'resettime': 1.0*24.0*60.0*60.0}

def use_defaults(name, val):
    if (val is not None) or (val is not np.nan):
        return val
    else:
        return defaults[name]

def ensure_shape(start, new):
    if isinstance(new, (int, float)):
        new = new*np.ones_like(start)
    return new

class Particles():

    def __init__(self, x, y, z, cellindices, verticaltreatment, dt=np.nan, zlevel=np.nan,
            indexlevel=np.nan, buoypart=np.nan, buoysurf=None,
            resettime=np.nan, xreset=np.nan, yreset=np.nan, zreset=np.nan, zlevelreset=np.nan):
        self.x = x
        self.y = y
        self.z = z
        self.verticaltreatment = ensure_shape(x, verticaltreatments[verticaltreatment])
        self.nparticles = len(x)

        self.dt = dt

        # 3D passive floats
        self.zlevel = ensure_shape(x, zlevel)

        # isopycnal floats
        self.buoypart = ensure_shape(x, buoypart)
        self.buoysurf = buoysurf
        self.cellindices = cellindices

        # index level following floats
        self.indexlevel = ensure_shape(x, indexlevel)

        # reset features
        self.resettime = resettime
        self.xreset = xreset
        self.yreset = yreset
        self.zreset = zreset
        self.zlevelreset = zlevelreset


class ParticleList():

    def __init__(self, particlelist):
        self.particlelist = particlelist

    def aggregate(self):
        self.len()

        # buoyancysurf
        buoysurf = np.array([])
        for alist in self.particlelist:
            buoysurf = np.unique(np.setxor1d(None,np.append(buoysurf, alist.buoysurf)))
        self.buoysurf = np.asarray(buoysurf, dtype='f8')


    def __getattr__(self, name):
        # __getattr__ ensures self.x is concatenated properly
        return self.concatenate(name)

    def concatenate(self, varname):
        var = getattr(self.particlelist[0], varname)
        for alist in self.particlelist[1:]:
            var = np.append(var, getattr(alist, varname))
        return var

    def append(particlelist):
        self.particlelist.append(particlelist[:])

    def len(self):
        self.nparticles = 0
        for alist in self.particlelist:
            self.nparticles += alist.nparticles

        return self.nparticles

    def write(self, f_name, f_decomp):

        self.aggregate()

        f_out = netCDF4.Dataset(f_name, 'w',format='NETCDF3_64BIT_OFFSET')

        f_out.createDimension('Time')
        f_out.createDimension('nParticles', self.nparticles)

        f_out.createVariable('xParticle', 'f8', ('Time','nParticles'))
        f_out.createVariable('yParticle', 'f8', ('Time','nParticles'))
        f_out.createVariable('zParticle', 'f8', ('Time','nParticles'))
        f_out.createVariable('zLevelParticle', 'f8', ('Time','nParticles'))
        f_out.createVariable('dtParticle', 'f8', ('Time','nParticles'))
        f_out.createVariable('buoyancyParticle', 'f8', ('Time','nParticles'))
        f_out.createVariable('currentBlock', 'i', ('Time', 'nParticles'))
        f_out.createVariable('currentCell', 'i', ('Time', 'nParticles'))
        f_out.createVariable('indexToParticleID', 'i', ('nParticles'))
        f_out.createVariable('verticalTreatment', 'i', ('Time','nParticles'))
        f_out.createVariable('indexLevel', 'i', ('Time','nParticles'))
        f_out.createVariable('resetTime', 'i', ('nParticles'))
        f_out.createVariable('currentBlockReset', 'i', ('nParticles'))
        f_out.createVariable('currentCellReset', 'i', ('nParticles'))
        f_out.createVariable('xParticleReset', 'f8', ('nParticles'))
        f_out.createVariable('yParticleReset', 'f8', ('nParticles'))
        f_out.createVariable('zParticleReset', 'f8', ('nParticles'))
        f_out.createVariable('zLevelParticleReset', 'f8', ('nParticles'))

        f_out.variables['xParticle'][0,:] = self.x
        f_out.variables['yParticle'][0,:] = self.y
        f_out.variables['zParticle'][0,:] = self.z

        f_out.variables['verticalTreatment'][0,:] = self.verticaltreatment

        f_out.variables['zLevelParticle'][0,:] = self.zlevel

        if len(self.buoysurf) > 0:
            f_out.createDimension('nBuoyancySurfaces', len(self.buoysurf))
            f_out.createVariable('buoyancySurfaceValues', 'f8', ('nBuoyancySurfaces'))
            f_out.variables['buoyancyParticle'][0,:] = self.buoypart
            f_out.variables['buoyancySurfaceValues'][:] = self.buoysurf

        f_out.variables['dtParticle'][0,:] = defaults['dt']
        # assume single-processor mode for now
        f_out.variables['currentBlock'][:] = 0
        f_out.variables['resetTime'][:] = defaults['resettime'] # reset each day
        f_out.variables['indexLevel'][:] = 1
        f_out.variables['indexToParticleID'][:] = np.arange(self.nparticles)

        # resets
        decomp = np.genfromtxt(f_decomp)
        f_out.variables['currentBlock'][0,:] = decomp[self.cellindices]
        f_out.variables['currentBlockReset'][:] = decomp[self.cellindices]
        f_out.variables['currentCell'][0,:] = -1
        f_out.variables['currentCellReset'][:] = -1
        f_out.variables['xParticleReset'][:] = f_out.variables['xParticle'][0,:]
        f_out.variables['yParticleReset'][:] = f_out.variables['yParticle'][0,:]
        f_out.variables['zParticleReset'][:] = f_out.variables['zParticle'][0,:]
        f_out.variables['zLevelParticleReset'][:] = f_out.variables['zLevelParticle'][0,:]

        f_out.close()

def get_cell_coords(f_init): #{{{
    return f_init.variables['xCell'][:], \
           f_init.variables['yCell'][:], \
           f_init.variables['zCell'][:] #}}}

def expand_nlevels(x, n): #{{{
    return np.tile(x, (n)) #}}}

def cell_centers(f_init): #{{{

    f_init = netCDF4.Dataset(f_init,'r')
    nparticles = len(f_init.dimensions['nCells'])
    xCell, yCell, zCell = get_cell_coords(f_init)
    f_init.close()

    return xCell, yCell, zCell  #}}}

def build_isopycnal_particles(buoysurf, f_init): #{{{

    xCell, yCell, zCell = cell_centers(f_init)
    nparticles = len(xCell)
    nbuoysurf = buoysurf.shape[0]

    x = expand_nlevels(xCell, nbuoysurf)
    y = expand_nlevels(yCell, nbuoysurf)
    z = expand_nlevels(zCell, nbuoysurf)

    buoypart = (np.tile(buoysurf,(nparticles,1))).reshape(nparticles*nbuoysurf,order='F').copy()
    cellindices = np.tile(np.arange(nparticles), (nbuoysurf))

    return Particles(x, y, z, cellindices, 'buoyancySurface', buoypart=buoypart, buoysurf=buoysurf) #}}}

def build_passive_floats(f_init, nvertlevels): #{{{

    xCell, yCell, zCell = cell_centers(f_init)
    x = expand_nlevels(xCell, nvertlevels)
    y = expand_nlevels(yCell, nvertlevels)
    z = expand_nlevels(zCell, nvertlevels)
    f_init = netCDF4.Dataset(f_init,'r')
    zlevel = -np.kron(np.linspace(0,1,nvertlevels+2)[1:-1], f_init.variables['bottomDepth'][:])
    cellindices = np.tile(np.arange(len(xCell)), (nvertlevels))
    f_init.close()


    return Particles(x, y, z, cellindices, 'passiveFloat', zlevel=zlevel) #}}}

def build_surface_floats(f_init):
    xCell, yCell, zCell = cell_centers(f_init)
    x = expand_nlevels(xCell, 1)
    y = expand_nlevels(yCell, 1)
    z = expand_nlevels(zCell, 1)
    cellindices = np.arange(len(xCell))

    return Particles(x, y, z, cellindices, 'indexLevel', indexlevel=1, zlevel=0)

def build_particle_file(f_init, f_name, f_decomp, types, buoySurf, nVertLevels):

    # build particles
    particlelist = []
    if 'buoyancy' in types:
        particlelist.append(build_isopycnal_particles(buoySurf, f_init))
    if 'passive' in types:
        particlelist.append(build_passive_floats(f_init, nVertLevels))
    if 'surface' in types:
        particlelist.append(build_surface_floats(f_init))

    # write particles to disk
    ParticleList(particlelist).write(f_name, f_decomp)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-i", "--init", dest="init",
            help="Name of netCDF init file",
            default="init.nc",
            metavar="PATH/INIT_NAME.nc")
    parser.add_argument("-g", "--graph", dest="graph",
            default='graph.info.part.',
            help="Path / name of graph file of form */*.info.part.",
            metavar="PATH/graph.info.part.")
    parser.add_argument("-o", "--particlefile", dest="particles",
            default="particles.nc",
            help="Path / name of netCDF particle file",
            metavar="PATH/particles.nc")
    parser.add_argument("-p", "--procs", dest="procs",
            help="Number of processors",
            metavar="INT")
    parser.add_argument("-t", "--types", dest="types",
            help="Types of particles",
            default="buoyancy",
            metavar="One of ['buoyancy', 'passive', 'surface', 'all']")
    parser.add_argument("--nvertlevels", dest="nvertlevels",
            default=10,
            help="Number of vertical levels for passive, 3D floats",
            metavar="INT")
    parser.add_argument("--nbuoysurf", dest="nbuoysurf",
            default=11,
            help="Number of buoyancy surfaces for isopycnally-constrained particles",
            metavar="INT")
    parser.add_argument("--potdensmin", dest="potdensmin",
            default=1028.5,
            help="Minimum value of potential density surface for isopycnally-constrained particles",
            metavar="INT")
    parser.add_argument("--potdensmax", dest="potdensmax",
            default=1030.0,
            help="Maximum value of potential density surface for isopycnally-constrained particles",
            metavar="INT")

    args = parser.parse_args()

    if not '.info.part.' in args.graph:
        OSError('Graph file processor count is inconsistent with processors specified!')
    if not ('.' + str(args.procs)) in args.graph:
        args.graph = args.graph + str(args.procs)

    if not os.path.exists(args.init):
        raise OSError('Init file {} not found.'.format(args.init))
    if not os.path.exists(args.graph):
        raise OSError('Graph file {} not found.'.format(args.graph))

    build_particle_file(args.init, args.particles, args.graph, args.types,
            np.linspace(args.potdensmin, args.potdensmax, int(args.nbuoysurf)), int(args.nvertlevels))

# vim: foldmethod=marker ai ts=4 sts=4 et sw=4 ft=python
