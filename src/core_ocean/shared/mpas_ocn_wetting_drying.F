! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_vmix
!
!> \brief MPAS ocean wetting and drying
!> \author Phillip J. Wolfram
!> \date   03/09/2018
!> \details
!>  This module contains the wetting and drying algorithms
!>  to prevent cell thicknesses from becoming too small.
!>
!
!-----------------------------------------------------------------------

module ocn_wetting_drying

   use mpas_kind_types
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_timer
   use mpas_io_units

   use ocn_constants
   use ocn_diagnostics
   use ocn_gm

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

   public :: ocn_wetting_drying_verify, ocn_prevent_drying

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------
   real(kind=RKIND), parameter :: eps = 1.0e-12_RKIND

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_wetting_drying_verify
!
!> \brief   Verifies that cells are not too dry.
!> \author  Phillip J. Wolfram
!> \date    03/09/2018
!> \details
!>  This routine checks that the minimum thickness in a cell is not
!>  too small.
!
!-----------------------------------------------------------------------

   subroutine ocn_wetting_drying_verify( block , minHeight, dt, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      type (block_type), intent(in) :: block !< block for computation
      real (kind=RKIND), intent(in) :: minHeight
      real (kind=RKIND), intent(in) :: dt !< Input: timestep

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

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

      type (mpas_pool_type), pointer :: statePool, meshPool, tendPool
      integer, dimension(:), pointer :: maxLevelCell
      integer, pointer :: nCellsSolve
      integer :: iCell, k
      integer :: debugUnit
      real (kind=RKIND), dimension(:,:), pointer :: layerThicknessCur, layerThicknessNew, layerThicknessTend
      real (kind=RKIND) :: minThickness, layerThick
      character (len=StrKIND) :: debugFilename

      !-----------------------------------------------------------------
      !
      ! call relevant routines for computing coefficients
      !
      !-----------------------------------------------------------------

      call mpas_pool_get_subpool(block % structs, 'state', statePool)
      call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
      call mpas_pool_get_subpool(block % structs, 'tend', tendPool)

      call mpas_pool_get_array(statePool, 'layerThickness', layerThicknessCur, timeLevel=1)
      call mpas_pool_get_array(statePool, 'layerThickness', layerThicknessNew, timeLevel=2)
      call mpas_pool_get_dimension(meshPool, 'nCellsSolve', nCellsSolve)
      call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
      call mpas_pool_get_array(tendPool, 'layerThickness', layerThicknessTend)

      err = 0

      call mpas_log_write( 'Verifying that cells are not dry...')

      ! check to make sure that there is no layer that is too dry
      minThickness = +1.0E34
      do iCell = 1, nCellsSolve
        do k = 1, maxLevelCell(iCell)
          ! this is artificially too restrictive, could have wetting and then drying over RK substeps, which could indicate
          ! drying when there is no drying occuring
          !layerThick = layerThicknessCur(k, iCell) + dt*layerThicknessTend(k, iCell)
          !call mpas_log_write( 'Layer thickness = $r', realArgs=(/ layerThick/))
          layerThick = layerThicknessNew(k, iCell)
          minThickness = min(minThickness, layerThick)
          if ( layerThick <  minHeight ) then
            ! report error if layerThickness constraint is violated
            if (err == 0) then
              debugFilename = ocn_build_log_filename('mpas_ocean_layerThickness_block_stats_', block % blockID)
              call mpas_new_unit(debugUnit)
              open(unit=debugUnit, file=debugFilename, form='formatted', status='unknown')
            end if
            err = iCell
            write(debugUnit, '(A, I5, A, I5, A, ES14.7, A)') 'ERROR: layerThickness too small at iCell=', iCell, &
                             ', k=', k, ' with thickness of ', layerThick , '.'
          end if
        end do
      end do

      call mpas_log_write('Minimum thickness is $r.', realArgs=(/minThickness/))

      if ( err > 0) then
        ! end the simulation if layerThickness constraint is violated
        call mpas_log_write( 'Done, some cells are have dried.')
        flush(debugUnit)
        close(debugUnit)
        call mpas_release_unit(debugUnit)
        call mpas_log_write( 'ERROR: Layer thickness smaller than $r, see ' // debugFilename , &
                             realArgs=(/ minHeight /), messageType=MPAS_LOG_CRIT)
      else
        call mpas_log_write( 'Done verifying that cells are wet.')
      end if

   end subroutine ocn_wetting_drying_verify !}}}

!***********************************************************************
!
!  routine ocn_prevent_drying
!
!> \brief   Prevents velocity tendency from causing cell drying
!> \author  Phillip J. Wolfram
!> \date    03/20/2018
!> \details
!>  This routine modifies velocity tendency to prevent cell drying.
!
!-----------------------------------------------------------------------
   subroutine ocn_prevent_drying(block, dt, rkWeight, config_zero_drying_velocity, err) !{{{

     !-----------------------------------------------------------------
     !
     ! input variables
     !
     !-----------------------------------------------------------------

     type (block_type), intent(in) :: block
     real (kind=RKIND), intent(in) :: dt
     real (kind=RKIND), intent(in) :: rkWeight
     logical, pointer :: config_zero_drying_velocity

     !-----------------------------------------------------------------
     !
     ! input/output variables
     !
     !-----------------------------------------------------------------

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


      type (mpas_pool_type), pointer :: tendPool, meshPool, statePool, provisStatePool, diagnosticsPool
      real (kind=RKIND), dimension(:, :), pointer :: layerThicknessCur, layerThicknessProvis, layerThicknessEdge
      real (kind=RKIND), dimension(:, :), pointer :: normalTransportVelocity, normalVelocity, wettingVelocity, vertAleTransportTop

      integer, dimension(:), pointer :: maxLevelEdgeTop
      integer, pointer :: nEdges
      integer :: iEdge, k

      err = 0
      !call mpas_log_write('Computing provisional variables...')

     call mpas_pool_get_subpool(block % structs, 'diagnostics', diagnosticsPool)
     call mpas_pool_get_subpool(block % structs, 'tend', tendPool)
     call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
     call mpas_pool_get_subpool(block % structs, 'state', statePool)
     call mpas_pool_get_subpool(block % structs, 'provis_state', provisStatePool)

     call mpas_pool_get_array(diagnosticsPool, 'vertAleTransportTop', vertAleTransportTop)
     call mpas_pool_get_array(diagnosticsPool, 'normalTransportVelocity', normalTransportVelocity)
     call mpas_pool_get_array(diagnosticsPool, 'layerThicknessEdge', layerThicknessEdge)
     call mpas_pool_get_array(diagnosticsPool, 'wettingVelocity', wettingVelocity)
     ! use thickness at n because constraint is h_n + dt*T_h > h_min
     call mpas_pool_get_array(statePool, 'layerThickness', layerThicknessCur, 1)
     call mpas_pool_get_array(provisStatePool, 'layerThickness', layerThicknessProvis, 1)
     !call mpas_pool_get_array(provisStatePool, 'normalVelocity', normalVelocity, 1)
     call mpas_pool_get_array(statePool, 'normalVelocity', normalVelocity, 1)

     call mpas_pool_get_dimension(block % dimensions, 'nEdges', nEdges)
     call mpas_pool_get_array(meshPool, 'maxLevelEdgeTop', maxLevelEdgeTop)

     !$omp do schedule(runtime)
     do iEdge = 1, nEdges
       wettingVelocity(:, iEdge) = 0.0_RKIND
     end do
     !$omp end do
     call mpas_threading_barrier()

     ! need estimate of tendency in order to 
     !call ocn_time_integrator_rk4_compute_thick_tends( block, dt, rk_substep_weights(rk_step), err )

     ! ensure cells stay wet by selectively damping cells with a damping tendency to make sure tendency doesn't dry cells
     call ocn_wetting_drying_wettingVelocity(meshPool, layerThicknessEdge, layerThicknessCur, layerThicknessProvis, &
                                             normalTransportVelocity, rkWeight, wettingVelocity, err)

     ! prevent drying from happening with selective wettingVelocity
     !$omp do schedule(runtime)
     do iEdge = 1, nEdges
       do k = 1, maxLevelEdgeTop(iEdge)
         if (abs(normalTransportVelocity(k,iEdge) + wettingVelocity(k,iEdge)) < eps)  then
           ! prevent spurious flux for close to zero values
           normalTransportVelocity(k, iEdge) = 0.0_RKIND
           normalVelocity(k, iEdge) = 0.0_RKIND
         else if (abs(normalTransportVelocity(k,iEdge) + wettingVelocity(k,iEdge)) <= abs(normalTransportVelocity(k,iEdge))) then
           normalTransportVelocity(k, iEdge) = normalTransportVelocity(k, iEdge) + wettingVelocity(k, iEdge)
           ! risk here-- if there is a GM bolus velocity this could cause problems and add in energy into the system
           normalVelocity(k, iEdge) = normalVelocity(k, iEdge) + wettingVelocity(k, iEdge)
         end if

         ! works as a fail-safe
         if (abs(wettingVelocity(k,iEdge)) > 0 .and. config_zero_drying_velocity) then
           normalTransportVelocity(k, iEdge) = 0.0_RKIND
           normalVelocity(k, iEdge) = 0.0_RKIND
         end if

       end do
       ! works but obviously is over-zealous
       !normalTransportVelocity(:, iEdge) = 0.0_RKIND ! normalTransportVelocity(:, iEdge) + wettingVelocity(:, iEdge)
       ! also limit normal velocity for consistency
       !normalVelocity(:, iEdge) = normalVelocity(:, iEdge) + wettingVelocity(:, iEdge)
     end do
     !$omp end do
     call mpas_threading_barrier()

   end subroutine ocn_prevent_drying !}}}

!***********************************************************************
!
!  routine ocn_wetting_drying_wettingVelocity
!
!> \brief   Computes velocity to prevent cell drying
!> \author  Phillip J. Wolfram
!> \date    03/19/2018
!> \details
!>  This routine adds wetting velocity opposed to drying motion
!>  to prevent cells from drying.
!
!-----------------------------------------------------------------------
   subroutine ocn_wetting_drying_wettingVelocity(meshPool, layerThicknessEdge, layerThicknessCur, layerThicknessProvis, &
       normalVelocity, dt, wettingVelocity, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      type (mpas_pool_type), intent(in) :: &
         meshPool           !< Input: horizonal mesh information

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         layerThicknessCur    !< Input: layer thickness at old time

       real (kind=RKIND), dimension(:,:), intent(in) :: &
         layerThicknessProvis    !< Input: provisional layer thickness

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         layerThicknessEdge     !< Input: layerThickness interpolated to an edge

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         normalVelocity     !< Input: transport

      real (kind=RKIND), intent(in) :: &
         dt     !< Input: time step

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(inout) :: &
         wettingVelocity          !< Input/Output: velocity wettingVelocityency

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

      integer :: iEdge, iCell, k, i, nCells
      integer, pointer :: nVertLevels
      integer, dimension(:), pointer :: nCellsArray
      integer, dimension(:), pointer :: nEdgesOnCell, nEdgesOnEdge, &
        maxLevelCell, maxLevelEdgeBot
      integer, dimension(:,:), pointer :: edgesOnCell, edgeSignOnCell

      real (kind=RKIND) :: divOutFlux, invAreaCell, layerThickness
      real (kind=RKIND), dimension(:), pointer :: dvEdge, areaCell

      real (kind=RKIND), pointer :: config_drying_min_cell_height

      err = 0

      call mpas_pool_get_config(ocnConfigs, 'config_drying_min_cell_height', config_drying_min_cell_height)

      call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)
      call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
      call mpas_pool_get_array(meshPool, 'edgesOnCell', edgesOnCell)
      call mpas_pool_get_array(meshPool, 'edgeSignOnCell', edgeSignOnCell)
      call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
      call mpas_pool_get_array(meshPool, 'maxLevelEdgeBot', maxLevelEdgeBot)
      call mpas_pool_get_array(meshPool, 'dvEdge', dvEdge)

      call mpas_pool_get_dimension(meshPool, 'nCellsArray', nCellsArray)
      call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)

      ! Only need to compute over 0 and 1 halos
      nCells = nCellsArray( 1 )

      call mpas_threading_barrier()

      !call mpas_log_write('IN WETTING AND DRYING $i', intArgs=(/nCells/))

      ! need predicted transport velocity to limit drying flux

      !! !$omp do schedule(runtime) private(invAreaCell, i, iEdge, k)
      do iCell = 1, nCells
        invAreaCell = 1.0_RKIND / areaCell(iCell)
        ! can switch with maxLevelEdgeBot(iEdge)
        do k = 1, maxLevelCell(iCell)
          divOutFlux = 0.0_RKIND
          !layerThickness = layerThicknessCur(k, iCell)
          !layerThickness = layerThicknessProvis(k, iCell)
          layerThickness = min(layerThicknessProvis(k, iCell), layerThicknessCur(k, iCell))
          !call mpas_log_write('prov = $r, curr=$r', realArgs=(/ layerThicknessProvis(k,iCell), layerThicknessCur(k,iCell)/))
          !layerThickness = layerThicknessCur(k, iCell)
          do i = 1, nEdgesOnCell(iCell)
            iEdge = edgesOnCell(i, iCell)
            if (k <= maxLevelEdgeBot(iEdge)) then
              ! only consider divergence flux leaving the cell
              !call mpas_log_write('u = $r, sign=$i', realArgs=(/ normalVelocity(k,iEdge)/), intArgs=(/edgesSignOnCell(i,iCell)/))
              if ( normalVelocity(k, iEdge) * edgeSignOnCell(i, iCell) < 0.0_RKIND ) then
                !call mpas_log_write('LIMIT FLUX1')
                ! potential problem here, if we have large flux out of one face but small out of the other, small flux could get
                ! lost due to truncation error
                divOutFlux = divOutFlux - normalVelocity(k, iEdge) * edgeSignOnCell(i, iCell) &
                  * layerThicknessEdge(k, iEdge) * dvEdge(iEdge)  * invAreaCell
              end if
            end if
          end do
          !call mpas_log_write('divOutFlux= $r, layerThickness = $r, dt= $r', realArgs=(/divOutFlux, layerThickness(k, iCell), dt/))

          divOutFlux = abs(divOutFlux)
          ! if layer thickness is too small, limit divergence flux outwards with opposite velocity
          if ((layerThickness - dt*divOutFlux ) <= (config_drying_min_cell_height))  then
            !call mpas_log_write('LIMIT FLUX')
            ! limit divOutFlux out of cell to keep it wet
            divOutFlux = (layerThickness - (config_drying_min_cell_height + eps)) / (dt*divOutFlux + eps)

            do i = 1, nEdgesOnCell(iCell)
              iEdge = edgesOnCell(i, iCell)
              if (k <= maxLevelEdgeBot(iEdge)) then
                if ( normalVelocity(k, iEdge) * edgeSignOnCell(i, iCell) < 0.0_RKIND ) then
                  ! each outgoing velocity is penalized (but not the incoming, wetting velocities)
                  !call mpas_log_write('theta = $r, $r', realArgs=(/ divOutFlux, min(max(0.0_RKIND, 1.0_RKIND - divOutFlux), 1.0_RKIND) /))
                  ! square the fractional term to make values near zero go to zero much quicker (to prevent threshold from being hit)
                  wettingVelocity(k, iEdge) =  - (min(max(0.0_RKIND, 1.0_RKIND - (divOutFlux*divOutFlux)), 1.0_RKIND)) * normalVelocity(k, iEdge)
                  ! just completely remove the velocity
                  !wettingVelocity(k, iEdge) =  - normalVelocity(k, iEdge)
                end if
              end if
            end do
          end if

        end do
      end do
      !! !$omp end do

      call mpas_threading_barrier()
      !call mpas_log_write('OUT OF WETTING AND DRYING')

   end subroutine ocn_wetting_drying_wettingVelocity !}}}

end module ocn_wetting_drying

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

! vim: foldmethod=marker
