! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_init_gaussian_wave
!
!> \brief MPAS ocean initialize case -- gaussian_wave
!> \author Phillip Wolfram
!> \date   8/23/18
!> \details
!>  This module contains the routines for initializing the
!>  gaussian_wave initial condition.
!>
!-----------------------------------------------------------------------

module ocn_init_gaussian_wave

   use mpas_kind_types
   use mpas_io_units
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants

   use ocn_constants
   use ocn_init_vertical_grids
   use ocn_init_cell_markers

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

   public :: ocn_init_setup_gaussian_wave, &
             ocn_init_validate_gaussian_wave

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_init_setup_gaussian_wave
!
!> \brief   Setup for this initial condition
!> \author  Phillip Wolfram
!> \date    8/23/18
!> \details
!>  This routine sets up the initial conditions for this case.
!
!-----------------------------------------------------------------------

  subroutine ocn_init_setup_gaussian_wave(domain, iErr)!{{{

    !--------------------------------------------------------------------

    type (domain_type), intent(inout) :: domain
    integer, intent(out) :: iErr

    type (block_type), pointer :: block_ptr
    type (mpas_pool_type), pointer :: meshPool
    type (mpas_pool_type), pointer :: statePool
    type (mpas_pool_type), pointer :: tracersPool
    type (mpas_pool_type), pointer :: verticalMeshPool
    type (mpas_pool_type), pointer :: diagnosticsPool

    ! local variables
    integer :: iCell, k, idx, iEdge, iVertex
    real (kind=RKIND) :: maxBottomDepth, dlon, dlat, dw
    real (kind=RKIND), dimension(:), pointer :: interfaceLocations
    real (kind=RKIND), parameter :: eps=1.0e-12
    logical :: wetting
    real (kind=RKIND) :: realbottom

    ! Define config variable pointers
    character (len=StrKIND), pointer :: config_init_configuration, config_vertical_grid
    integer, pointer :: config_gaussian_wave_vert_levels
    real (kind=RKIND), pointer :: config_gaussian_wave_lon_center, config_gaussian_wave_lat_center, &
      config_gaussian_wave_width, config_gaussian_wave_amplitude, config_gaussian_wave_slr_amp
    logical, pointer :: config_use_wetting_drying
    real (kind=RKIND), pointer :: config_drying_min_cell_height

    ! Define dimension pointers
    integer, pointer :: nCellsSolve, nEdgesSolve, nVerticesSolve, nVertLevels, nVertLevelsP1
    integer, pointer :: index_temperature, index_salinity

    ! Define variable pointers
    logical, pointer :: on_a_sphere
    integer, dimension(:), pointer :: maxLevelCell
    real (kind=RKIND), dimension(:), pointer :: ssh
    real (kind=RKIND), dimension(:), pointer :: xCell, yCell,refBottomDepth, refZMid, &
         vertCoordMovementWeights, bottomDepth, &
         fCell, fEdge, fVertex, dcEdge, &
         lonCell, latCell, latEdge, latVertex
    real (kind=RKIND), dimension(:,:), pointer :: layerThickness, restingThickness, zMid
    real (kind=RKIND), dimension(:,:,:), pointer :: activeTracers

    iErr = 0

    call mpas_pool_get_config(ocnConfigs, 'config_init_configuration', config_init_configuration)

    if(config_init_configuration .ne. trim('gaussian_wave')) return

    ! Get config flag settings

    call mpas_pool_get_config(ocnConfigs, 'config_vertical_grid', config_vertical_grid)

    call mpas_pool_get_config(ocnConfigs, 'config_gaussian_wave_vert_levels', config_gaussian_wave_vert_levels)
    call mpas_pool_get_config(ocnConfigs, 'config_gaussian_wave_lon_center', config_gaussian_wave_lon_center)
    call mpas_pool_get_config(ocnConfigs, 'config_gaussian_wave_lat_center', config_gaussian_wave_lat_center)
    call mpas_pool_get_config(ocnConfigs, 'config_gaussian_wave_width', config_gaussian_wave_width)
    call mpas_pool_get_config(ocnConfigs, 'config_gaussian_wave_amplitude', config_gaussian_wave_amplitude)
    call mpas_pool_get_config(ocnConfigs, 'config_gaussian_wave_slr_amp', config_gaussian_wave_slr_amp)

    ! wetting and drying parameters
    call mpas_pool_get_config(ocnConfigs, 'config_drying_min_cell_height', config_drying_min_cell_height)
    call mpas_pool_get_config(ocnConfigs, 'config_use_wetting_drying', config_use_wetting_drying)

    ! Determine vertical grid for configuration
    call mpas_pool_get_subpool(domain % blocklist % structs, 'mesh', meshPool)
    call mpas_pool_get_array(meshPool, 'refBottomDepth', refBottomDepth)
    call mpas_pool_get_array(meshPool, 'bottomDepth', bottomDepth)
    call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
    call mpas_pool_get_dimension(meshPool, 'nVertLevelsP1', nVertLevelsP1)
    call mpas_pool_get_config(meshPool, 'on_a_sphere', on_a_sphere)

    ! you may restrict your case geometry as follows:
    if ( .not. on_a_sphere ) call mpas_log_write('The gaussian_wave configuration can only be applied ' &
      // 'to a spherical mesh. Exiting...', MPAS_LOG_CRIT)

    ! assign config variables
    ! ! Set refBottomDepth and refZMid
    maxBottomDepth = 60.0_RKIND
    nVertLevels  = config_gaussian_wave_vert_levels
    nVertLevelsP1 = nVertLevels + 1


    allocate(interfaceLocations(nVertLevelsP1))
    call ocn_generate_vertical_grid( config_vertical_grid, interfaceLocations, ocnConfigs )

    !--------------------------------------------------------------------
    ! Use this section to set initial values
    !--------------------------------------------------------------------

    block_ptr => domain % blocklist
    do while(associated(block_ptr))
       call mpas_pool_get_subpool(block_ptr % structs, 'mesh', meshPool)
       call mpas_pool_get_subpool(block_ptr % structs, 'state', statePool)
       call mpas_pool_get_subpool(block_ptr % structs, 'verticalMesh', verticalMeshPool)
       call mpas_pool_get_subpool(block_ptr % structs, 'diagnostics', diagnosticsPool)
       call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)

       call mpas_pool_get_dimension(meshPool, 'nCellsSolve', nCellsSolve)
       call mpas_pool_get_dimension(meshPool, 'nEdgesSolve', nEdgesSolve)
       call mpas_pool_get_dimension(meshPool, 'nVerticesSolve', nVerticesSolve)

       call mpas_pool_get_dimension(tracersPool, 'index_temperature', index_temperature)
       call mpas_pool_get_dimension(tracersPool, 'index_salinity', index_salinity)

       call mpas_pool_get_array(meshPool, 'vertCoordMovementWeights', vertCoordMovementWeights)
       call mpas_pool_get_array(meshPool, 'bottomDepth', bottomDepth)
       call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
       call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
       call mpas_pool_get_array(meshPool, 'latCell', latCell)
       call mpas_pool_get_array(meshPool, 'latEdge', latEdge)
       call mpas_pool_get_array(meshPool, 'latVertex', latVertex)
       call mpas_pool_get_array(meshPool, 'fCell', fCell)
       call mpas_pool_get_array(meshPool, 'fEdge', fEdge)
       call mpas_pool_get_array(meshPool, 'fVertex', fVertex)

       call mpas_pool_get_array(tracersPool, 'activeTracers', activeTracers, 1)
       call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, 1)
       call mpas_pool_get_array(diagnosticsPool, 'zMid', zMid)
       call mpas_pool_get_array(statePool, 'ssh', ssh, 1)

       call mpas_pool_get_array(verticalMeshPool, 'refZMid', refZMid)
       call mpas_pool_get_array(verticalMeshPool, 'restingThickness', restingThickness)

       do k = 1, nVertLevels
          refBottomDepth(k) = maxBottomDepth* interfaceLocations(k+1)
          refZMid(k) = - 0.5_RKIND * (interfaceLocations(k+1) + interfaceLocations(k)) * maxBottomDepth
       end do

       ! Set vertCoordMovementWeights
       vertCoordMovementWeights(:) = 1.0_RKIND

       do iCell = 1, nCellsSolve

          ! Set ssh
          !call mpas_log_write('bottomDepth $r= ', realArgs=(/bottomDepth(iCell)/))
          dlat = latCell(iCell) - config_gaussian_wave_lat_center*pii/180.0_RKIND
          dlon = abs(lonCell(iCell) - config_gaussian_wave_lon_center*pii/180.0_RKIND)
          dw = config_gaussian_wave_width*pii/180.0_RKIND
          if (dlon > pii) dlon = dlon - 2.0_RKIND**pii
          ssh(iCell) = config_gaussian_wave_slr_amp + &
                       config_gaussian_wave_amplitude * exp(-(dlat**2.0_RKIND + dlon**2.0_RKIND) / &
                                                              (2.0_RKIND*dw**2.0_RKIND))

          !! handle dry cell case to get needed layer numbers
          !wetting = .false.
          !if (config_use_wetting_drying) then
          !  if (bottomDepth(iCell) < 0) then
          !    wetting = .true.
          !  end if
          !end if

          ! add in offset to keep elevations below 0.0
          !bottomDepth(iCell) = bottomDepth(iCell) + 20.0_RKIND

          !bottomDepth(iCell) = 10.0_RKIND + 10.0_RKIND* exp(-(dlat**2.0_RKIND + dlon**2.0_RKIND) / &
          !                                  (2.0_RKIND*0.1_RKIND**2.0_RKIND))

          !do k = 1, maxLevelCell(iCell)
          !  restingThickness(k, iCell) = 1.0_RKIND
          !  layerThickness(k, iCell) = restingThickness(k, iCell)
          !end do

          !! keep wet cells for min(maxLevelCell) = 3  (current limitation in
          !! code: https://github.com/MPAS-Dev/MPAS/issues/1169
          !if (bottomDepth(iCell) < refBottomDepth(3)) then
          !  if (wetting) then
          !    realbottom = bottomDepth(iCell)
          !  end if
          !  bottomDepth(iCell) = refBottomDepth(3)
          !end if

          ! initialize with constant layer thickness of 3 m globally
          maxLevelCell(iCell) = 3

          bottomDepth(iCell) = min(maxBottomDepth, max(-10.0_RKIND, bottomDepth(iCell)))

          if (.not. config_use_wetting_drying) then
            bottomDepth(iCell) = max(3.0_RKIND, bottomDepth(iCell))
          end if

          ! make sure depth is thick enough via ssh = TOTAL_DEPTH - bottomDepth
          ssh(iCell) = max(ssh(iCell) + bottomDepth(iCell), &
            maxLevelCell(iCell)*(config_drying_min_cell_height + eps)) &
            - bottomDepth(iCell)

          !bottomDepth(iCell) = bottomDepth(iCell) + 12.0_RKIND
          !ssh(iCell) = ssh(iCell) - 12.0_RKIND

          ! add in ssh perterbation
          do k=1, maxLevelCell(iCell)
            !layerThickness(k,iCell) = 1.0_RKIND
            layerThickness(k,iCell) = max(config_drying_min_cell_height + eps, &
              1.0_RKIND/float(maxLevelCell(iCell))*(ssh(iCell) + bottomDepth(iCell)))
            restingThickness(:, iCell) = layerThickness(:, iCell)
          end do

          ! Set temperature
          idx = index_temperature
          do k = 1, nVertLevels
             activeTracers(idx, k, iCell) = 20.0_RKIND
          end do

          ! Set salinity
          idx = index_salinity
          do k = 1, nVertLevels
             activeTracers(idx, k, iCell) = 33.0_RKIND
          end do

          !maxLevelCell(iCell) = -1
          !do k = 1, nVertLevels
          !  if (refBottomDepth(k) >= bottomDepth(iCell)) then
          !    maxLevelCell(iCell) = k
          !    !bottomDepth(iCell) = refBottomDepth(k)
          !    exit
          !  end if
          !end do

          !if (maxLevelCell(iCell) == -1) then
          !  maxLevelCell(iCell) = nVertLevels
          !  bottomDepth(iCell) = refBottomDepth( nVertLevels )
          !end if

          !! Set layerThickness and restingThickness
          !call ocn_compute_layerThickness_zMid_from_bottomDepth(layerThickness(:,iCell), zMid(:,iCell), &
          !  refBottomDepth, bottomDepth(iCell), maxLevelCell(iCell), nVertLevels, iErr, &
          !  restingThickness=restingThickness(:,iCell), &
          !  ssh=ssh(iCell))
          !restingThickness(:, iCell) = layerThickness(:, iCell)
          ! Set layerThickness and restingThickness
          ! Uniform layer thickness across lat/lon

          !! build initial, unperturbed vertical grid
          !restingThickness(1, iCell) = refBottomDepth(1)
          !layerThickness(1, iCell) = restingThickness(1, iCell)
          !do k = 2, maxLevelCell(iCell)
          !  restingThickness(k, iCell) = refBottomDepth(k) - refBottomDepth(k-1)
          !  layerThickness(k, iCell) = restingThickness(k, iCell)
          !end do

          !! handle wetting / drying
          !if (config_use_wetting_drying) then
          !  !if ( wetting ) then
          !    ! thin out layers to start with "dry" condition
          !    do k=1, maxLevelCell(iCell)
          !      restingThickness(k, iCell) = 2.0_RKIND*config_drying_min_cell_height + eps
          !      layerThickness(k, iCell) = restingThickness(k, iCell)
          !      !bottomDepth(iCell) = max(realbottom, -3.1_RKIND)
          !    end do

          !  !end if
          !  bottomDepth(iCell) = bottomDepth(iCell)
          !end if
       end do


       ! Set Coriolis parameters
       do iCell = 1, nCellsSolve
         fCell(iCell) = 2.0_RKIND * omega * sin(latCell(iCell))
       end do
       do iEdge = 1, nEdgesSolve
         fEdge(iEdge) = 2.0_RKIND * omega * sin(latEdge(iEdge))
       end do
       do iVertex = 1, nVerticesSolve
         fVertex(iVertex) = 2.0_RKIND * omega * sin(latVertex(iVertex))
       end do


       block_ptr => block_ptr % next
    end do

    deallocate(interfaceLocations)
    !--------------------------------------------------------------------

  end subroutine ocn_init_setup_gaussian_wave!}}}

!***********************************************************************
!
!  routine ocn_init_validate_gaussian_wave
!
!> \brief   Validation for this initial condition
!> \author  Phillip Wolfram
!> \date    8/23/18
!> \details
!>  This routine validates the configuration options for this case.
!
!-----------------------------------------------------------------------

   subroutine ocn_init_validate_gaussian_wave(configPool, packagePool, iocontext, iErr)!{{{

   !--------------------------------------------------------------------
      type (mpas_pool_type), intent(inout) :: configPool, packagePool
      type (mpas_io_context_type), intent(inout) :: iocontext

      integer, intent(out) :: iErr

      character (len=StrKIND), pointer :: config_init_configuration
      integer, pointer :: config_vert_levels, config_gaussian_wave_vert_levels

      iErr = 0

      call mpas_pool_get_config(configPool, 'config_init_configuration', config_init_configuration)

      if(config_init_configuration .ne. trim('gaussian_wave')) return

      call mpas_pool_get_config(configPool, 'config_vert_levels', config_vert_levels)
      call mpas_pool_get_config(configPool, 'config_gaussian_wave_vert_levels', config_gaussian_wave_vert_levels)

      if(config_vert_levels <= 0 .and. config_gaussian_wave_vert_levels > 0) then
         config_vert_levels = config_gaussian_wave_vert_levels
      else if (config_vert_levels <= 0) then
         call mpas_log_write( 'Validation failed for gaussian_wave. Not given a usable value for vertical levels.', MPAS_LOG_CRIT)
         iErr = 1
      end if

   !--------------------------------------------------------------------

   end subroutine ocn_init_validate_gaussian_wave!}}}


!***********************************************************************

end module ocn_init_gaussian_wave

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
