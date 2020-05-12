!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_vmix_gotm
!
!> \brief MPAS ocean vertical mixing interface to GOTM
!> \author Qing Li
!> \date   11 May 2020
!> \details
!>  This module contains the routines for calls into GOTM
!>
!
!-----------------------------------------------------------------------

module ocn_vmix_gotm

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_timer
   use mpas_constants
   use mpas_log

   use ocn_constants

   ! TODO: read in GOTM namelist for testing <20200512, Qing Li> !
   use mpas_io_units

   use turbulence, only: init_turbulence, do_turbulence
   use turbulence, only: gotm_vertVisc=>num, gotm_vertDiff=>nuh
   use mtridiagonal, only: init_tridiagonal

   implicit none
   private
   save

   !--------------------------------------------------------------------
   !
   ! Public parameters
   !
   !--------------------------------------------------------------------

   !--------------------------------------------------------------------
   !
   ! Public member functions
   !
   !--------------------------------------------------------------------

   public :: ocn_vmix_coefs_gotm_build, &
             ocn_vmix_gotm_init

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

   logical :: gotmOn


!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_vmix_coefs_cmvix_build
!
!> \brief   Computes mixing coefficients using GOTM
!> \author  Qing Li
!> \date    11 May 2020
!> \details
!>  This routine computes the vertical mixing coefficients for momentum
!>  and tracers by calling GOTM routines.
!
!-----------------------------------------------------------------------

   subroutine ocn_vmix_coefs_gotm_build(meshPool, statePool, forcingPool, diagnosticsPool, err, timeLevelIn)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      type (mpas_pool_type), intent(in) :: &
         meshPool          !< Input: mesh information

      integer, intent(in), optional :: timeLevelIn !< Input: time level for state pool

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      type (mpas_pool_type), intent(inout) :: &
         statePool         !< Input/Output: state information

      type (mpas_pool_type), intent(inout) :: &
         diagnosticsPool   !< Input/Output: diagnostic information

      type (mpas_pool_type), intent(inout) :: &
         forcingPool   !< Input/Output: forcing information

      !-----------------------------------------------------------------
      !
      ! output variables
      !
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      !-----------------------------------------------------------------
      !
      ! local variables
      !
      !-----------------------------------------------------------------

      integer, dimension(:), pointer :: &
        maxLevelCell, nEdgesOnCell

      real (kind=RKIND), dimension(:), pointer :: &
        latCell, lonCell, bottomDepth, surfaceBuoyancyForcing, surfaceFrictionVelocity, fCell, &
        boundaryLayerDepth, ssh, indexBoundaryLayerDepth, dcEdge, dvEdge, areaCell, iceFraction, &
        boundaryLayerDepthSmooth, windSpeed10m

      real (kind=RKIND), dimension(:,:), pointer :: &
        vertViscTopOfCell, vertDiffTopOfCell, layerThickness, &
        zMid, zTop, density, displacedDensity, potentialDensity, &
        bulkRichardsonNumber, RiTopOfCell, BruntVaisalaFreqTop, &
        bulkRichardsonNumberBuoy, bulkRichardsonNumberShear, unresolvedShear, normalVelocity

      real (kind=RKIND), dimension(:,:,:), pointer :: vertNonLocalFlux
      integer, pointer :: index_vertNonLocalFluxTemp
      integer, dimension(:,:), pointer :: edgesOnCell, cellsOnCell, cellMask

      integer :: k, i, iCell, jCell, iNeighbor, iter, timeLevel, kIndexOBL, kav, iEdge, nCells
      integer :: edgeCount, nEdges, topIndex, nsmooth, kpp_stage
      integer, pointer :: nVertLevels, nVertLevelsP1
      integer, dimension(:), pointer :: nCellsArray
      integer, dimension(:), allocatable :: surfaceAverageIndex

      real (kind=RKIND) :: r, layerSum, bulkRichardsonNumberStop, sfc_layer_depth, invAreaCell
      real (kind=RKIND) :: normalVelocityAv, factor, delU2, areaSum, blTemp
      real (kind=RKIND) :: sigma, turbulentScalarVelocityScalePoint
      real (kind=RKIND), dimension(:), allocatable :: Nsqr_iface, turbulentScalarVelocityScale, &
                                                      deltaVelocitySquared, normalVelocitySum, &
                                                      potentialDensitySum, RiTemp
      real (kind=RKIND), dimension(:), allocatable, target :: RiSmoothed, BVFSmoothed, OBLDepths, interfaceForcings
      logical :: bulkRichardsonFlag

      !-----------------------------------------------------------------
      !
      ! call relevant routines for computing mixing-related fields
      ! note that the user can choose multiple options and the
      !   mixing fields have to be added/merged together
      !
      !-----------------------------------------------------------------

      !
      ! assume no errors during initialization and set to 1 when error is encountered
      !
      err=0

      if (present(timeLevelIn)) then
         timeLevel = timeLevelIn
      else
         timeLevel = 1
      end if

      !
      ! only build up viscosity/diffusivity if GOTM is turned on
      !
      if ( .not. gotmOn ) return

      !
      ! set parameters
      !
      ! call mpas_pool_get_config(ocnConfigs, 'config_cvmix_background_viscosity', config_cvmix_background_viscosity)
      call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
      call mpas_pool_get_dimension(meshPool, 'nVertLevelsP1', nVertLevelsP1)
      call mpas_pool_get_dimension(meshPool, 'nCellsArray', nCellsArray)
      call mpas_pool_get_array(meshPool, 'dcEdge', dcEdge)
      call mpas_pool_get_array(meshPool, 'dvEdge', dvEdge)
      call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)
      call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
      call mpas_pool_get_array(statePool, 'normalVelocity', normalVelocity, timeLevel)
      call mpas_pool_get_array(meshPool, 'edgesOnCell', edgesOnCell)
      call mpas_pool_get_array(meshPool, 'cellsOnCell', cellsOnCell)
      call mpas_pool_get_array(meshPool, 'cellMask', cellMask)

      !
      ! set pointers for fields related to position on sphere
      !
      call mpas_pool_get_array(meshPool, 'latCell', latCell)
      call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
      call mpas_pool_get_array(meshPool, 'fCell', fCell)

      !
      ! set pointers for fields related to vertical mesh
      !
      call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
      call mpas_pool_get_array(meshPool, 'bottomDepth', bottomDepth)

      call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, timeLevel)
      call mpas_pool_get_array(statePool, 'ssh', ssh, timeLevel)

      call mpas_pool_get_array(diagnosticsPool, 'zTop', zTop)
      call mpas_pool_get_array(diagnosticsPool, 'zMid', zMid)

      !
      ! set pointers for fields related ocean state
      !
      call mpas_pool_get_array(diagnosticsPool, 'density', density)
      call mpas_pool_get_array(diagnosticsPool, 'displacedDensity', displacedDensity)
      call mpas_pool_get_array(diagnosticsPool, 'potentialDensity', potentialDensity)
      call mpas_pool_get_array(diagnosticsPool, 'bulkRichardsonNumber', bulkRichardsonNumber)
      call mpas_pool_get_array(diagnosticsPool, 'unresolvedShear', unresolvedShear)
      call mpas_pool_get_array(diagnosticsPool, 'boundaryLayerDepth', boundaryLayerDepth)
      call mpas_pool_get_array(diagnosticsPool, 'boundaryLayerDepthSmooth', boundaryLayerDepthSmooth)
      call mpas_pool_get_array(diagnosticsPool, 'RiTopOfCell', RiTopOfCell)
      call mpas_pool_get_array(diagnosticsPool, 'BruntVaisalaFreqTop',BruntVaisalaFreqTop)
      call mpas_pool_get_array(diagnosticsPool, 'bulkRichardsonNumberBuoy',bulkRichardsonNumberBuoy)
      call mpas_pool_get_array(diagnosticsPool, 'bulkRichardsonNumberShear',bulkRichardsonNumberShear)
      call mpas_pool_get_array(diagnosticsPool, 'indexBoundaryLayerDepth',indexBoundaryLayerDepth)

      !
      ! set pointers for fields related to ocean forcing state
      !
      call mpas_pool_get_array(forcingPool, 'iceFraction', iceFraction)
      call mpas_pool_get_array(forcingPool, 'windSpeed10m', windSpeed10m)

      !
      ! set pointers for fields related forcing at ocean surface
      !
      call mpas_pool_get_array(diagnosticsPool, 'surfaceFrictionVelocity', surfaceFrictionVelocity)
      call mpas_pool_get_array(diagnosticsPool, 'surfaceBuoyancyForcing', surfaceBuoyancyForcing)

      !
      ! set pointers for viscosity/diffusivity and initialize to zero
      !
      call mpas_pool_get_array(diagnosticsPool, 'vertViscTopOfCell', vertViscTopOfCell)
      call mpas_pool_get_array(diagnosticsPool, 'vertDiffTopOfCell', vertDiffTopOfCell)


      !
      ! set pointers for nonlocal flux and initialize to zero
      !
      call mpas_pool_get_array(diagnosticsPool, 'vertNonLocalFlux', vertNonLocalFlux)
      call mpas_pool_get_dimension(diagnosticsPool, 'index_vertNonLocalFluxTemp', index_vertNonLocalFluxTemp)

      nCells = nCellsArray( size(nCellsArray) )

      !$omp do schedule(runtime)
      do iCell = 1, nCells
         vertViscTopOfCell(:, iCell) = 0.0_RKIND
         vertDiffTopOfCell(:, iCell) = 0.0_RKIND
         vertNonLocalFlux(:, :, iCell) = 0.0_RKIND
      end do
      !$omp end do

      nCells = nCellsArray( 3 )


      allocate(OBLDepths(nVertLevels))
      allocate(interfaceForcings(nVertLevels))

      allocate(Nsqr_iface(nVertLevels+1))
      allocate(turbulentScalarVelocityScale(nVertLevels))
      allocate(RiSmoothed(nVertLevels+1))
      allocate(BVFSmoothed(nVertLevels+1))
      allocate(RiTemp(nVertLevels+1))

      allocate(normalVelocitySum(nVertLevels))
      allocate(potentialDensitySum(nVertLevels))
      allocate(surfaceAverageIndex(nVertLevels))
      allocate(deltaVelocitySquared(nVertLevels))

      do k = 1, nVertLevels
         Nsqr_iface(k) = 0.0_RKIND
         turbulentScalarVelocityScale(k) = 0.0_RKIND
      end do
      Nsqr_iface(nVertLevelsP1) = 0.0_RKIND

      call mpas_timer_start('gotm cell loop', .false.)
      ! TODO: do_turbulence here <20200511, Qing Li> !
      call mpas_timer_stop('gotm cell loop')

      deallocate(Nsqr_iface)
      deallocate(turbulentScalarVelocityScale)
      deallocate(RiSmoothed)
      deallocate(RiTemp)
      deallocate(BVFSmoothed)
      deallocate(normalVelocitySum)
      deallocate(potentialDensitySum)
      deallocate(surfaceAverageIndex)
      deallocate(deltaVelocitySquared)

      deallocate(OBLDepths)
      deallocate(interfaceForcings)

   !--------------------------------------------------------------------

   end subroutine ocn_vmix_coefs_gotm_build!}}}

!***********************************************************************
!
!  routine ocn_vmix_gotm_init
!
!> \brief   Initializes ocean vertical mixing quantities using GOTM
!> \author  Qing Li
!> \date    11 May 2020
!> \details
!>  This routine initializes a variety of quantities related to
!>  vertical mixing in the ocean. Parameters are set by calling into GOTM
!
!-----------------------------------------------------------------------


   subroutine ocn_vmix_gotm_init(domain,err)!{{{

   !--------------------------------------------------------------------

      !-----------------------------------------------------------------
      !
      ! call individual init routines for each parameterization
      !
      !-----------------------------------------------------------------

      implicit none

      type (domain_type), intent(inout) :: domain !< Input/Output: domain information

      integer, intent(out) :: err !< Output: error flag

      ! TODO: read in namelist for testing <20200512, Qing Li> !
      integer :: namlstUnit !< unit for GOTM namelist file

      integer, pointer :: nVertLevels
      type (block_type), pointer :: block

      ! GOTM
      logical, pointer :: config_use_gotm

      !
      ! assume no errors during initialization and set to 1 when error is encountered
      !
      err=0

      call mpas_pool_get_config(ocnConfigs, 'config_use_gotm', config_use_gotm)
      gotmOn = config_use_gotm

      !
      ! only initialize if GOTM is turned on
      !
      if (.not. gotmOn) return

      !
      ! When GOTM is turned on, all other vertical mixing schemes should be off
      ! Test to make sure this is the case.
      !
      ! test here, err=1 if a problem

      !
      ! pull nVertLevels out of the mesh structure
      !
      call mpas_pool_get_dimension(domain % blocklist % dimensions, 'nVertLevels', nVertLevels)

      !
      ! initialize GOTM turbulence module
      ! TODO: read in namelist for now, will need to modify init_turbulence if not reading namelist from this subroutine? <20200512, Qing Li> !
      call mpas_new_unit(namlstUnit)
      call init_turbulence(namlstUnit, 'gotmturb.nml', nVertLevels)
      call mpas_release_unit(namlstUnit)

      ! initialize GOTM tridiagnoal solver
      call init_tridiagonal(nVertLevels)

   !--------------------------------------------------------------------

   end subroutine ocn_vmix_gotm_init!}}}

!***********************************************************************

end module ocn_vmix_gotm

!||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
