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
   use turbulence, only: gotm_vertVisc=>num, gotm_vertDiffT=>nuh, &
                         gotm_vertDiffS=>nus, gotm_TKE=>tke,      &
                         gotm_eps=>eps, gotm_L=>L
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

   subroutine ocn_vmix_coefs_gotm_build(dt, meshPool, statePool, forcingPool, diagnosticsPool, err, timeLevelIn)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), intent(in) :: dt !< Input: MPAS-O timestep
      type (mpas_pool_type), intent(in) :: &
         meshPool          !< Input: mesh information

      integer, intent(in), optional ::     &
         timeLevelIn !< Input: time level for state pool

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

      logical, save :: first = .true.
      integer, pointer :: nVertLevels, nVertLevelsP1

      integer, dimension(:), pointer :: &
        maxLevelCell, nEdgesOnCell, nCellsArray

      integer, dimension(:,:), pointer :: edgesOnCell

      real (kind=RKIND), dimension(:), pointer :: &
        bottomDepth, surfaceBuoyancyForcing, surfaceFrictionVelocity, &
        dcEdge, dvEdge, areaCell

      real (kind=RKIND), dimension(:,:), pointer :: &
        vertViscTopOfCell, vertDiffTopOfCell, layerThickness, &
        BruntVaisalaFreqTop, normalVelocity, zMid, &
        gotmVertViscTopOfCellCur, gotmVertDiffTopOfCellCur, &
        gotmTKETopOfCellCur, gotmDissTopOfCellCur, gotmLengthTopOfCellCur, &
        gotmVertViscTopOfCellNew, gotmVertDiffTopOfCellNew, &
        gotmTKETopOfCellNew, gotmDissTopOfCellNew, gotmLengthTopOfCellNew

      real (kind=RKIND), dimension(:,:,:), pointer :: vertNonLocalFlux

      integer :: i, iCell, iEdge, k, ki, timeLevel, nCells

      real (kind=RKIND) :: invAreaCell, factor, delU2
      real (kind=RKIND), dimension(:), allocatable :: &
        shearSquaredTop, NSquaredTop, hLayer

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

      !
      ! set pointers for fields related to vertical mesh
      !
      call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
      call mpas_pool_get_array(meshPool, 'bottomDepth', bottomDepth)

      call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, timeLevel)

      !
      ! set pointers for fields related ocean state
      !
      call mpas_pool_get_array(diagnosticsPool, 'BruntVaisalaFreqTop',BruntVaisalaFreqTop)

      !
      ! set pointers for fields related forcing at ocean surface
      !
      call mpas_pool_get_array(diagnosticsPool, 'surfaceFrictionVelocity', surfaceFrictionVelocity)
      call mpas_pool_get_array(diagnosticsPool, 'surfaceBuoyancyForcing', surfaceBuoyancyForcing)

      !
      ! set pointer for z at the midpoint of vertical grid
      !
      call mpas_pool_get_array(diagnosticsPool, 'zMid', zMid)

      !
      ! set pointers for viscosity/diffusivity
      !
      call mpas_pool_get_array(diagnosticsPool, 'vertViscTopOfCell', vertViscTopOfCell)
      call mpas_pool_get_array(diagnosticsPool, 'vertDiffTopOfCell', vertDiffTopOfCell)

      !
      ! set pointers for nonlocal flux
      !
      call mpas_pool_get_array(diagnosticsPool, 'vertNonLocalFlux', vertNonLocalFlux)

      !
      ! set pointers for GOTM viscosity/diffusivity, TKE, rate of dissipation and turbulent length scale
      ! (current and new state)
      !
      call mpas_pool_get_array(statePool, 'gotmVertViscTopOfCell', gotmVertViscTopOfCellCur, 1)
      call mpas_pool_get_array(statePool, 'gotmVertViscTopOfCell', gotmVertViscTopOfCellNew, 2)
      call mpas_pool_get_array(statePool, 'gotmVertDiffTopOfCell', gotmVertDiffTopOfCellCur, 1)
      call mpas_pool_get_array(statePool, 'gotmVertDiffTopOfCell', gotmVertDiffTopOfCellNew, 2)
      call mpas_pool_get_array(statePool, 'gotmTKETopOfCell', gotmTKETopOfCellCur, 1)
      call mpas_pool_get_array(statePool, 'gotmTKETopOfCell', gotmTKETopOfCellNew, 2)
      call mpas_pool_get_array(statePool, 'gotmDissTopOfCell', gotmDissTopOfCellCur, 1)
      call mpas_pool_get_array(statePool, 'gotmDissTopOfCell', gotmDissTopOfCellNew, 2)
      call mpas_pool_get_array(statePool, 'gotmLengthTopOfCell', gotmLengthTopOfCellCur, 1)
      call mpas_pool_get_array(statePool, 'gotmLengthTopOfCell', gotmLengthTopOfCellNew, 2)

      nCells = nCellsArray( size(nCellsArray) )

      !$omp do schedule(runtime)
      do iCell = 1, nCells
         vertViscTopOfCell(:, iCell) = 0.0_RKIND
         vertDiffTopOfCell(:, iCell) = 0.0_RKIND
         vertNonLocalFlux(:, :, iCell) = 0.0_RKIND
      end do
      !$omp end do

      ! initialize GOTM arrays at the first step
      ! MPAS-O (1:nlev+1) <-> GOTM (nlev:0)
      if (first) then
         !$omp do schedule(runtime)
         do iCell = 1, nCells
            do k = 1, nVertLevelsP1
               ki = nVertLevelsP1 - k
               gotmVertViscTopOfCellCur(k, iCell) = gotm_vertVisc(ki)
               gotmVertDiffTopOfCellCur(k, iCell) = gotm_vertDiffT(ki)
               gotmTKETopOfCellCur(k, iCell) = gotm_TKE(ki)
               gotmDissTopOfCellCur(k, iCell) = gotm_eps(ki)
               gotmLengthTopOfCellCur(k, iCell) = gotm_L(ki)
               gotmVertViscTopOfCellNew(k, iCell) = gotm_vertVisc(ki)
               gotmVertDiffTopOfCellNew(k, iCell) = gotm_vertDiffT(ki)
               gotmTKETopOfCellNew(k, iCell) = gotm_TKE(ki)
               gotmDissTopOfCellNew(k, iCell) = gotm_eps(ki)
               gotmLengthTopOfCellNew(k, iCell) = gotm_L(ki)
            end do
         end do
         !$omp end do
         first = .false.
      end if

      nCells = nCellsArray( 3 )

      ! allocate squared shear and buoyancy frequencies
      allocate(shearSquaredTop(nVertLevelsP1))
      allocate(NSquaredTop(nVertLevelsP1))
      allocate(hLayer(nVertLevelsP1))

      call mpas_timer_start('gotm cell loop', .false.)

      !$omp do schedule(runtime)
      do iCell = 1, nCells

         ! initialize arrays
         shearSquaredTop = 0.0_RKIND
         NSquaredTop = 0.0_RKIND
         hLayer = 0.0_RKIND

         ! compute shear frequency squared
         ! TODO: use KE conserving shear following Burchard (2002)? <20200514, Qing Li> !
         invAreaCell = 1.0_RKIND / areaCell(iCell)
         do k=2,maxLevelCell(iCell)
            do i = 1, nEdgesOnCell(iCell)
               iEdge = edgesOnCell(i, iCell)
               factor = 0.5_RKIND * dcEdge(iEdge) * dvEdge(iEdge) * invAreaCell
               delU2 = (normalVelocity(k-1,iEdge) - normalVelocity(k,iEdge))**2
               shearSquaredTop(k) = shearSquaredTop(k) + factor * delU2
            enddo
            shearSquaredTop(k) = shearSquaredTop(k) / (zMid(k-1,iCell) - zMid(k,iCell))**2
         end do
         shearSquaredTop(1) = shearSquaredTop(2)
         shearSquaredTop(maxLevelCell(iCell)+1) = shearSquaredTop(maxLevelCell(iCell))

         ! fill arrays of the squared buoyancy frequency and layer thickness
         do k = 1, maxLevelCell(iCell)
            NSquaredTop(k) = BruntVaisalaFreqTop(k,iCell)
            hLayer(k) = layerThickness(k,iCell)
         enddo

         ! fill the bottom values -- GOTM requires size of nlev+1 for both arrays
         NSquaredTop(maxLevelCell(iCell)+1) = NSquaredTop(maxLevelCell(iCell))
         hLayer(maxLevelCell(iCell)+1) = hLayer(maxLevelCell(iCell))

         ! fill GOTM arrays with current values at iCell
         do k = 0, maxLevelCell(iCell)
            ki = maxLevelCell(iCell) - k + 1
            gotm_vertVisc(k) = gotmVertViscTopOfCellCur(ki, iCell)
            gotm_vertDiffT(k) = gotmVertDiffTopOfCellCur(ki, iCell)
            gotm_vertDiffS(k) = gotmVertDiffTopOfCellCur(ki, iCell)
            gotm_TKE(k) = gotmTKETopOfCellCur(ki, iCell)
            gotm_eps(k) = gotmDissTopOfCellCur(ki, iCell)
            gotm_L(k) = gotmLengthTopOfCellCur(ki, iCell)
         end do

         !
         ! interface with do_turbulence():
         !
         !   subroutine do_turbulence(nlev,dt,depth,u_taus,u_taub,z0s,z0b,h,NN,SS,xP)
         !
         !   nlev          : number of vertical layers
         !   dt            : time step (s)
         !   depth         : water depth (m)
         !   u_taus        : surface friction velocity (m/s)
         !   u_taub        : bottom friction velocity (m/s)
         !   z0s           : surface roughness length (m)
         !   z0b           : bottom roughness length (m)
         !   h             : layer thickness (m)
         !   NN            : bouyancy frequency squared (1/s^2)
         !   SS            : shear-frequency squared (1/s^2)
         !   xP (optional) : extra TKE production due to, e.g., seagrass (m^2/s^3)
         !
         ! MPAS-O (1:nlev+1) <-> GOTM (nlev:0)
         !
         call do_turbulence(maxLevelCell(iCell),                                 &
                            dt,                                                  &
                            bottomDepth(iCell),                                  &
                            surfaceFrictionVelocity(iCell),                      &
                            0.0_RKIND,                                           &
                            0.02_RKIND,                                          &
                            0.0015_RKIND,                                        &
                            hLayer(maxLevelCell(iCell)+1:1:-1),                  &
                            NSquaredTop(maxLevelCell(iCell)+1:1:-1),             &
                            shearSquaredTop(maxLevelCell(iCell)+1:1:-1))
         ! TODO: use zero bottom friction and predefined surface/bottom roughness length for testing (may be a lower estimate) <20200512, Qing Li> !
         ! TODO: use MPAS-O dt for testing, though subcycling is possible <20200512, Qing Li> !

         ! update GOTM arrays for next time step
         ! MPAS-O (1:nlev+1) <-> GOTM (nlev:0)
         do k = 1, maxLevelCell(iCell)
            ki = maxLevelCell(iCell) - k + 1
            gotmVertViscTopOfCellNew(k, iCell) = gotm_vertVisc(ki)
            gotmVertDiffTopOfCellNew(k, iCell) = gotm_vertDiffT(ki)
            gotmTKETopOfCellNew(k, iCell) = gotm_TKE(ki)
            gotmDissTopOfCellNew(k, iCell) = gotm_eps(ki)
            gotmLengthTopOfCellNew(k, iCell) = gotm_L(ki)
            ! update viscosity/diffusivity
            vertViscTopOfCell(k, iCell) = gotm_vertVisc(ki)
            vertDiffTopOfCell(k, iCell) = gotm_vertDiffT(ki)
         end do

      end do
      !$omp end do

      call mpas_timer_stop('gotm cell loop')

      deallocate(shearSquaredTop)
      deallocate(NSquaredTop)
      deallocate(hLayer)

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
