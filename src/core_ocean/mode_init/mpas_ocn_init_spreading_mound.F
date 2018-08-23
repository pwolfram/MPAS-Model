! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_init_spreading_mound
!
!> \brief MPAS ocean initialize case -- spreading_mound
!> \author Phillip J. Wolfram
!> \date   12/4/18
!> \details
!>  This module contains the routines for initializing the
!>  spreading_mound initial condition
!>
!>  In order to add a new analysis member, do the following:
!>  1. In src/core_ocean/mode_init, copy these to your new analysis member name:
!>     cp mpas_ocn_init_spreading_mound.F mpas_ocn_init_your_new_name.F
!>     cp Registry_spreading_mound.xml Registry_ocn_your_new_name.xml
!>
!>  2. In those two new files, replace the following text:
!>     spreading_mound, Phillip J. Wolfram, 12/4/18
!>     spreading_mound uses underscores (subroutine names), like your_new_name.
!>
!>  3. Add a #include line for your registry to
!>     src/core_ocean/mode_init/Registry.xml
!>
!>  4. Copy and change spreading_mound lines in src/core_ocean/mode_init/mpas_ocn_init_mode.F
!>
!>  5. Add these dependency lines by following spreading_mound examples in:
!>     in src/core_ocean/mode_init/Makefile
!
!-----------------------------------------------------------------------

module ocn_init_spreading_mound

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

   public :: ocn_init_setup_spreading_mound, &
             ocn_init_validate_spreading_mound

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_init_setup_spreading_mound
!
!> \brief   Setup for this initial condition
!> \author  Phillip J. Wolfram
!> \date    12/4/18
!> \details
!>  This routine sets up the initial conditions for this case.
!
!-----------------------------------------------------------------------

  subroutine ocn_init_setup_spreading_mound(domain, iErr)!{{{

    !--------------------------------------------------------------------

    type (domain_type), intent(inout) :: domain
    integer, intent(out) :: iErr

    type (block_type), pointer :: block_ptr
    type (mpas_pool_type), pointer :: meshPool
    type (mpas_pool_type), pointer :: statePool
    type (mpas_pool_type), pointer :: tracersPool
    type (mpas_pool_type), pointer :: verticalMeshPool

    ! local variables
    integer :: iCell, k, idx
    real (kind=RKIND) :: yMin, yMax, xMin, xMax, dcEdgeMin, dcEdgeMinGlobal
    real (kind=RKIND) :: yMinGlobal, yMaxGlobal, yMidGlobal, xMinGlobal, xMaxGlobal
    real (kind=RKIND) :: localVar1, localVar2
    real (kind=RKIND), dimension(:), pointer :: interfaceLocations
    real (kind=RKIND) :: Tscale, dx, dy

    ! Define config variable pointers
    character (len=StrKIND), pointer :: config_init_configuration, config_vertical_grid
    real (kind=RKIND), pointer :: config_spreading_mound_eta0
    real (kind=RKIND), pointer :: config_spreading_mound_R0
    real (kind=RKIND), pointer :: config_spreading_mound_xc
    real (kind=RKIND), pointer :: config_spreading_mound_yc
    real (kind=RKIND), pointer :: config_spreading_mound_coriolis_parameter
    integer, pointer :: config_spreading_mound_vert_levels
    logical, pointer :: config_use_wetting_drying
    real (kind=RKIND), pointer :: config_drying_min_cell_height
    real (kind=RKIND), parameter :: eps=1.0e-12

    ! Define dimension pointers
    integer, pointer :: nCellsSolve, nEdgesSolve, nVertLevels, nVertLevelsP1
    integer, pointer :: index_temperature, index_salinity

    ! Define variable pointers
    logical, pointer :: on_a_sphere
    integer, dimension(:), pointer :: maxLevelCell
    real (kind=RKIND), dimension(:), pointer :: ssh
    real (kind=RKIND), dimension(:), pointer :: xCell, yCell,refBottomDepth, refZMid, &
         vertCoordMovementWeights, bottomDepth, &
         fCell, fEdge, fVertex, dcEdge
    real (kind=RKIND), dimension(:,:), pointer :: layerThickness, restingThickness
    real (kind=RKIND), dimension(:,:,:), pointer :: activeTracers

    iErr = 0

    call mpas_pool_get_config(ocnConfigs, 'config_init_configuration', config_init_configuration)

    if(config_init_configuration .ne. trim('spreading_mound')) return

    ! Get config flag settings

    call mpas_pool_get_config(ocnConfigs, 'config_vertical_grid', config_vertical_grid)

    call mpas_pool_get_config(ocnConfigs, 'config_spreading_mound_eta0', config_spreading_mound_eta0)
    call mpas_pool_get_config(ocnConfigs, 'config_spreading_mound_R0', config_spreading_mound_R0)
    call mpas_pool_get_config(ocnConfigs, 'config_spreading_mound_xc', config_spreading_mound_xc)
    call mpas_pool_get_config(ocnConfigs, 'config_spreading_mound_yc', config_spreading_mound_yc)
    call mpas_pool_get_config(ocnConfigs, 'config_spreading_mound_vert_levels', config_spreading_mound_vert_levels)
    call mpas_pool_get_config(ocnConfigs, 'config_spreading_mound_coriolis_parameter', &
                                           config_spreading_mound_coriolis_parameter)
    
    ! wetting and drying parameters
    call mpas_pool_get_config(ocnConfigs, 'config_drying_min_cell_height', config_drying_min_cell_height)
    call mpas_pool_get_config(ocnConfigs, 'config_use_wetting_drying', config_use_wetting_drying)

    ! Determine vertical grid for configuration
    call mpas_pool_get_subpool(domain % blocklist % structs, 'mesh', meshPool)
    call mpas_pool_get_dimension(meshPool, 'nVertLevelsP1', nVertLevelsP1)
    call mpas_pool_get_config(meshPool, 'on_a_sphere', on_a_sphere)

    ! you may restrict your case geometry as follows:
    if ( on_a_sphere ) call mpas_log_write('The spreading_mound configuration can only be applied ' &
        // 'to a planar mesh. Exiting...', MPAS_LOG_CRIT)

    if ( .not. config_use_wetting_drying ) call mpas_log_write('The spreading_mound configuration can only be applied ' &
        // 'with wetting and drying. Exiting...', MPAS_LOG_CRIT)

    nVertLevelsP1 = config_spreading_mound_vert_levels + 1
    allocate(interfaceLocations(nVertLevelsP1))
    call ocn_generate_vertical_grid( config_vertical_grid, interfaceLocations, ocnConfigs)

    !--------------------------------------------------------------------
    ! Use this section to set initial values
    !--------------------------------------------------------------------

    block_ptr => domain % blocklist
    do while(associated(block_ptr))
       call mpas_pool_get_subpool(block_ptr % structs, 'mesh', meshPool)
       call mpas_pool_get_subpool(block_ptr % structs, 'state', statePool)
       call mpas_pool_get_subpool(block_ptr % structs, 'verticalMesh', verticalMeshPool)
       call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)

       call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
       call mpas_pool_get_dimension(meshPool, 'nCellsSolve', nCellsSolve)

       call mpas_pool_get_dimension(tracersPool, 'index_temperature', index_temperature)
       call mpas_pool_get_dimension(tracersPool, 'index_salinity', index_salinity)

       call mpas_pool_get_array(meshPool, 'xCell', xCell)
       call mpas_pool_get_array(meshPool, 'yCell', yCell)
       call mpas_pool_get_array(meshPool, 'refBottomDepth', refBottomDepth)
       call mpas_pool_get_array(meshPool, 'vertCoordMovementWeights', vertCoordMovementWeights)
       call mpas_pool_get_array(meshPool, 'bottomDepth', bottomDepth)
       call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
       call mpas_pool_get_array(meshPool, 'fCell', fCell)
       call mpas_pool_get_array(meshPool, 'fEdge', fEdge)
       call mpas_pool_get_array(meshPool, 'fVertex', fVertex)

       call mpas_pool_get_array(tracersPool, 'activeTracers', activeTracers, 1)
       call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, 1)
       call mpas_pool_get_array(statePool, 'ssh', ssh, 1)

       call mpas_pool_get_array(verticalMeshPool, 'refZMid', refZMid)
       call mpas_pool_get_array(verticalMeshPool, 'restingThickness', restingThickness)


       ! ! Set refBottomDepth and refZMid
       do k = 1, nVertLevels
          ! will make the top ssh 0, others below (assumes bottom depth is max intial height of mound)
          refBottomDepth(k) = config_spreading_mound_eta0 * interfaceLocations(k+1)
          refZMid(k) = - 0.5_RKIND * (interfaceLocations(k+1) + interfaceLocations(k)) * config_spreading_mound_eta0
       end do

       ! Set vertCoordMovementWeights
       vertCoordMovementWeights(:) = 1.0_RKIND

       Tscale = 1.0_RKIND/sqrt(2.0_RKIND*gravity*config_spreading_mound_eta0)

       do iCell = 1, nCellsSolve
          
          ! Set bottomDepth
          bottomDepth(iCell) = config_spreading_mound_eta0
          
          ! set ssh
          dx = xCell(iCell) - config_spreading_mound_xc
          dy = yCell(iCell) - config_spreading_mound_yc
          
          ssh(iCell) = config_spreading_mound_eta0 * &
            (1.0_RKIND - (dx*dx + dy*dy)/(config_spreading_mound_R0*config_spreading_mound_R0))
          
          if ( ssh(iCell) < (config_spreading_mound_vert_levels*config_drying_min_cell_height) ) then
            ssh(iCell) = config_spreading_mound_vert_levels*config_drying_min_cell_height + eps
          end if

          ! reorient for coordinate system in vertical
          ssh(iCell) = - config_spreading_mound_eta0 + ssh(iCell)

          ! Set maxLevelCell
          maxLevelCell(iCell) = config_spreading_mound_vert_levels
          
          ! add in ssh perterbation
          do k=1, maxLevelCell(iCell)
            layerThickness(k,iCell) = max(config_drying_min_cell_height + eps, &
              1.0_RKIND/float(maxLevelCell(iCell))*(ssh(iCell) + bottomDepth(iCell)))
            restingThickness(:, iCell) = layerThickness(:, iCell)
          end do

          ! Set temperature
          idx = index_temperature
          do k = 1, nVertLevels
             activeTracers(idx, k, iCell) = 0.0_RKIND
          end do

          ! Set salinity
          idx = index_salinity
          do k = 1, nVertLevels
             activeTracers(idx, k, iCell) = 0.0_RKIND
          end do

          !! Set Coriolis parameters, if other than zero
          !fCell(iCell) = config_spreading_mound_coriolis_parameter
          !fEdge(iCell) = config_spreading_mound_coriolis_parameter
          !fVertex(iCell) = config_spreading_mound_coriolis_parameter

       end do

       block_ptr => block_ptr % next
    end do

    deallocate(interfaceLocations)
    !--------------------------------------------------------------------

  end subroutine ocn_init_setup_spreading_mound!}}}

!***********************************************************************
!
!  routine ocn_init_validate_spreading_mound
!
!> \brief   Validation for this initial condition
!> \author  Phillip J. Wolfram
!> \date    12/4/18
!> \details
!>  This routine validates the configuration options for this case.
!
!-----------------------------------------------------------------------

   subroutine ocn_init_validate_spreading_mound(configPool, packagePool, iocontext, iErr)!{{{

   !--------------------------------------------------------------------
      type (mpas_pool_type), intent(inout) :: configPool, packagePool
      type (mpas_io_context_type), intent(inout) :: iocontext

      integer, intent(out) :: iErr

      character (len=StrKIND), pointer :: config_init_configuration
      integer, pointer :: config_vert_levels, config_spreading_mound_vert_levels

      iErr = 0

      call mpas_pool_get_config(configPool, 'config_init_configuration', config_init_configuration)

      if(config_init_configuration .ne. trim('spreading_mound')) return

      call mpas_pool_get_config(configPool, 'config_vert_levels', config_vert_levels)
      call mpas_pool_get_config(configPool, 'config_spreading_mound_vert_levels', config_spreading_mound_vert_levels)

      if(config_vert_levels <= 0 .and. config_spreading_mound_vert_levels > 0) then
         config_vert_levels = config_spreading_mound_vert_levels
      else if (config_vert_levels <= 0) then
         call mpas_log_write( 'Validation failed for spreading_mound. Not given a usable value for vertical levels.', MPAS_LOG_CRIT)
         iErr = 1
      end if

   !--------------------------------------------------------------------

   end subroutine ocn_init_validate_spreading_mound!}}}


!***********************************************************************

end module ocn_init_spreading_mound

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
