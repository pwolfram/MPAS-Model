#!/usr/bin/env python
"""
File writes a particle input dataset for use in MPAS-O / E3SM.

Example usage is

    python make_particle_file.py -i init.nc -g graph.info.part.6 \
            -o particles.nc -p 6 --spatialfilter SouthernOceanXYZ

Phillip J. Wolfram
Last Modified: 08/03/2018
"""

import netCDF4
import numpy as np
from scipy import spatial
from scipy import sparse
from pyamg.classical import split

verticaltreatments = {'indexLevel':1, 'fixedZLevel': 2, 'passiveFloat': 3, 'buoyancySurface': 4, 'argoFloat': 5}
defaults = {'dt': 300, 'resettime': 1.0*24.0*60.0*60.0}


def use_defaults(name, val): #{{{
    if (val is not None) or (val is not np.nan):
        return val
    else:
        return defaults[name] #}}}


def ensure_shape(start, new): #{{{
    if isinstance(new, (int, float)):
        new = new*np.ones_like(start)
    return new #}}}


def southern_ocean_only_xyz(x, y, z, maxNorth=-45.0): #{{{
    sq = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arcsin(z / sq)
    #lon = np.arctan2(y, x)
    ok = np.pi/180.0*maxNorth
    ids = lat < ok
    return ids #}}}


def southern_ocean_only_planar(x, y, z, maxy=1000.0*1e3): #{{{
    ids = y < maxy
    return ids #}}}


def remap_particles(fin, fpart, fdecomp): #{{{
    """
    Remap particles onto a new grid decomposition.

    Load in particle positions, locations of grid cell centers, and decomposition
    corresponding to fin.

    The goal is to update particle field currentBlock to comply with the new grid
    as defined by fin and decomp.  NOTE: FIN AND FDECOMP MUST BE COMPATIBLE!

    We assume that all particles will be within the domain such that a nearest
    neighbor search is sufficient to make the remap.

    Phillip Wolfram
    LANL
    Origin: 08/19/2014, Updated: 07/13/2018
    """
    # load the files
    f_in = netCDF4.Dataset(fin, 'r')
    f_part = netCDF4.Dataset(fpart,'r+')

    # get the particle data
    xpart = f_part.variables['xParticle']
    ypart = f_part.variables['yParticle']
    zpart = f_part.variables['zParticle']
    currentBlock = f_part.variables['currentBlock']
    try:
        currentCell = f_part.variables['currentCell']
    except:
        currentCell = f_part.createVariable('currentCell', 'i', ('nParticles'))

    # get the cell positions
    xcell = f_in.variables['xCell']
    ycell = f_in.variables['yCell']
    zcell = f_in.variables['zCell']

    # build the spatial tree
    tree = spatial.cKDTree(np.vstack((xcell,ycell,zcell)).T)

    # get nearest cell for each particle
    dvEdge = f_in.variables['dvEdge']
    maxdist = 2.0*max(dvEdge[:])
    _, cellIndices = tree.query(np.vstack((xpart,ypart,zpart)).T,distance_upper_bound=maxdist,k=1)

    # load the decomposition (apply to latest time step)
    decomp = np.genfromtxt(fdecomp)
    currentBlock[-1,:] = decomp[cellIndices]
    currentCell[-1,:] = cellIndices + 1

    # close the files
    f_in.close()
    f_part.close()

    #}}}


def downsample_points(x, y, z, tri): #{{{
    """
    Downsample points using algebraic multigrid splitting.

    Note, currently assumes that all points on grid are equidistant, which does
    a numeric (not area-weighted) downsampling.

    Phillip Wolfram
    LANL
    Origin: 03/09/2015, Updated: 08/03/2018
    """
    # build A
    Np = x.shape[0]
    A = sparse.lil_matrix((Np, Np))
    for anum in np.arange(Np):
        adj = np.where(np.sum(anum == tri, axis=1))
        for atri in np.sort(adj):
            A[anum, tri[atri,:]] += np.ones_like(atri)[:,np.newaxis]
        A[anum, anum] *= -1
    # A = -A

    # Grab root-nodes (i.e., Coarse / Fine splitting)
    Cpts = np.asarray(split.PMIS(A.tocsr()), dtype=bool)

    return x[Cpts], y[Cpts], z[Cpts] #}}}

class Particles(): #{{{

    def __init__(self, x, y, z, cellindices, verticaltreatment, dt=np.nan, zlevel=np.nan,
            indexlevel=np.nan, buoypart=np.nan, buoysurf=None, spatialfilter=None,
            resettime=np.nan, xreset=np.nan, yreset=np.nan, zreset=np.nan, zlevelreset=np.nan): #{{{

        # start with all the indicies and restrict
        ids = np.arange(len(x))
        if spatialfilter:
            if 'SouthernOceanXYZ' in spatialfilter:
                ids = np.intersect1d(ids, np.arange(len(x))[southern_ocean_only_xyz(x,y,z)])
            if 'SouthernOceanXY' in spatialfilter:
                ids = np.intersect1d(ids, np.arange(len(x))[southern_ocean_only_planar(x,y,z)])

        self.x = x[ids]
        self.y = y[ids]
        self.z = z[ids]
        self.verticaltreatment = ensure_shape(self.x, verticaltreatments[verticaltreatment])
        self.nparticles = len(self.x)

        self.dt = dt

        # 3D passive floats
        self.zlevel = ensure_shape(x, zlevel)[ids]

        # isopycnal floats
        if buoysurf is not None:
            self.buoysurf = buoysurf
        self.buoypart = ensure_shape(x, buoypart)[ids]
        self.cellindices = cellindices[ids]

        # index level following floats
        self.indexlevel = ensure_shape(x, indexlevel)[ids]

        # reset features
        self.resettime = ensure_shape(x, resettime)[ids]
        self.xreset = ensure_shape(x, xreset)[ids]
        self.yreset = ensure_shape(x, yreset)[ids]
        self.zreset = ensure_shape(x, zreset)[ids]
        self.zlevelreset = ensure_shape(x, zlevelreset)[ids]

        return #}}}
#}}}


class ParticleList(): #{{{

    def __init__(self, particlelist): #{{{
        self.particlelist = particlelist #}}}


    def aggregate(self): #{{{
        self.len()

        # buoyancysurf
        buoysurf = np.array([])
        for alist in self.particlelist:
            if 'buoysurf' in dir(alist):
                buoysurf = np.unique(np.setdiff1d(np.append(buoysurf, alist.buoysurf), None))
        if len(buoysurf) > 0:
            self.buoysurf = np.asarray(buoysurf, dtype='f8')
        else:
            self.buoysurf = None

        return #}}}


    def __getattr__(self, name): #{{{
        # __getattr__ ensures self.x is concatenated properly
        return self.concatenate(name) #}}}


    def concatenate(self, varname): #{{{
        var = getattr(self.particlelist[0], varname)
        for alist in self.particlelist[1:]:
            var = np.append(var, getattr(alist, varname))
        return var #}}}


    def append(particlelist): #{{{
        self.particlelist.append(particlelist[:]) #}}}


    def len(self): #{{{
        self.nparticles = 0
        for alist in self.particlelist:
            self.nparticles += alist.nparticles

        return self.nparticles #}}}


    def write(self, f_name, f_decomp):  #{{{

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
        return #}}}
#}}}


def get_cell_coords(f_init): #{{{
    return f_init.variables['xCell'][:], \
           f_init.variables['yCell'][:], \
           f_init.variables['zCell'][:] #}}}


def expand_nlevels(x, n): #{{{
    return np.tile(x, (n)) #}}}


def cell_centers(f_init, downsample): #{{{

    f_init = netCDF4.Dataset(f_init,'r')
    nparticles = len(f_init.dimensions['nCells'])
    xCell, yCell, zCell = get_cell_coords(f_init)
    if downsample:
        tri = f_init.variables['cellsOnVertex'][:,:] - 1
        xCell, yCell, zCell = downsample_points(xCell, yCell, zCell, tri)
    f_init.close()

    return xCell, yCell, zCell  #}}}


def build_isopycnal_particles(xCell, yCell, zCell, buoysurf, afilter): #{{{

    nparticles = len(xCell)
    nbuoysurf = buoysurf.shape[0]

    x = expand_nlevels(xCell, nbuoysurf)
    y = expand_nlevels(yCell, nbuoysurf)
    z = expand_nlevels(zCell, nbuoysurf)

    buoypart = (np.tile(buoysurf,(nparticles,1))).reshape(nparticles*nbuoysurf,order='F').copy()
    cellindices = np.tile(np.arange(nparticles), (nbuoysurf))

    return Particles(x, y, z, cellindices, 'buoyancySurface', buoypart=buoypart, buoysurf=buoysurf, spatialfilter=afilter) #}}}


def build_passive_floats(xCell, yCell, zCell, f_init, nvertlevels, afilter): #{{{

    x = expand_nlevels(xCell, nvertlevels)
    y = expand_nlevels(yCell, nvertlevels)
    z = expand_nlevels(zCell, nvertlevels)
    f_init = netCDF4.Dataset(f_init,'r')
    zlevel = -np.kron(np.linspace(0,1,nvertlevels+2)[1:-1], f_init.variables['bottomDepth'][:])
    cellindices = np.tile(np.arange(len(xCell)), (nvertlevels))
    f_init.close()


    return Particles(x, y, z, cellindices, 'passiveFloat', zlevel=zlevel, spatialfilter=afilter) #}}}


def build_surface_floats(xCell, yCell, zCell): #{{{

    x = expand_nlevels(xCell, 1)
    y = expand_nlevels(yCell, 1)
    z = expand_nlevels(zCell, 1)
    cellindices = np.arange(len(xCell))

    return Particles(x, y, z, cellindices, 'indexLevel', indexlevel=1, zlevel=0) #}}}


def build_particle_file(f_init, f_name, f_decomp, types, spatialfilter, buoySurf, nVertLevels, downsample): #{{{

    xCell, yCell, zCell = cell_centers(f_init, downsample)

    # build particles
    particlelist = []
    if 'buoyancy' in types or 'all' in types:
        particlelist.append(build_isopycnal_particles(xCell, yCell, zCell, buoySurf, spatialfilter))
    if 'passive' in types or 'all' in types:
        particlelist.append(build_passive_floats(xCell, yCell, zCell, f_init, nVertLevels, spatialfilter ))
    # apply surface particles everywhere to ensure that LIGHT works
    # (allow for some load-imbalance for filters)
    if 'surface' in types or 'all' in types:
        particlelist.append(build_surface_floats(xCell, yCell, zCell))

    # write particles to disk
    ParticleList(particlelist).write(f_name, f_decomp)

    return #}}}


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
            metavar="One or more of ['buoyancy', 'passive', 'surface', 'all']")
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
    parser.add_argument("--spatialfilter", dest="spatialfilter",
            default=None,
            help="Apply a certain type of spatial filter, e.g., 'SouthernOcean'",
            metavar="STRING")
    parser.add_argument("--remap", dest="remap",
            action="store_true",
            help="Remap particle file based on input mesh and decomposition.")
    parser.add_argument("--downsample", dest="downsample",
            action="store_true",
            help="Downsample particle positions using AMG.")

    args = parser.parse_args()

    if not '.info.part.' in args.graph:
        OSError('Graph file processor count is inconsistent with processors specified!')
    if not ('.' + str(args.procs)) in args.graph:
        args.graph = args.graph + str(args.procs)

    if not os.path.exists(args.init):
        raise OSError('Init file {} not found.'.format(args.init))
    if not os.path.exists(args.graph):
        raise OSError('Graph file {} not found.'.format(args.graph))

    if not args.remap:
        print('Building particle file...')
        build_particle_file(args.init, args.particles, args.graph, args.types, args.spatialfilter,
                np.linspace(args.potdensmin, args.potdensmax, int(args.nbuoysurf)), int(args.nvertlevels),
                args.downsample)
        print('Done building particle file')
    else:
        print('Remapping particles...')
        remap_particles(args.init, args.particles, args.graph)
        print('Done remapping particles')

# vim: foldmethod=marker ai ts=4 sts=4 et sw=4 ft=python
