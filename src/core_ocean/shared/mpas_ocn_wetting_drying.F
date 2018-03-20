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

   subroutine ocn_wetting_drying_verify( block , minHeight, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      type (block_type), intent(in) :: block !< block for computation
      real (kind=RKIND), intent(in) :: minHeight

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

      type (mpas_pool_type), pointer :: statePool, meshPool
      integer, dimension(:), pointer :: maxLevelCell
      integer, pointer :: nCellsSolve
      integer :: iCell, k
      integer :: debugUnit
      real (kind=RKIND), dimension(:,:), pointer :: layerThicknessCur
      real (kind=RKIND) :: minThickness
      character (len=StrKIND) :: debugFilename

      !-----------------------------------------------------------------
      !
      ! call relevant routines for computing coefficients
      !
      !-----------------------------------------------------------------

      err = 0

      call mpas_pool_get_subpool(block % structs, 'state', statePool)
      call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)

      call mpas_pool_get_array(statePool, 'layerThickness', layerThicknessCur, timeLevel=1)
      call mpas_pool_get_dimension(meshPool, 'nCellsSolve', nCellsSolve)
      call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)

      err = 0

      call mpas_log_write( 'Verifying that cells are not dry...')

      ! check to make sure that there is no layer that is too dry
      minThickness = +1.0E34
      do iCell = 1, nCellsSolve
        do k = 1, maxLevelCell(iCell)
          !call mpas_log_write( 'Layer thickness = $r', realArgs=(/ layerThicknessCur(k, iCell)/))
          minThickness = min(minThickness, layerThicknessCur(k, iCell))
          if ( layerThicknessCur(k, iCell) <  minHeight ) then
            ! report error if layerThickness constraint is violated
            if (err == 0) then
              debugFilename = ocn_build_log_filename('mpas_ocean_layerThickness_block_stats_', block % blockID)
              call mpas_new_unit(debugUnit)
              open(unit=debugUnit, file=debugFilename, form='formatted', status='unknown')
            end if
            err = iCell
            write(debugUnit, '(A, I5, A, I5, A, ES14.7, A)') 'ERROR: layerThickness too small at iCell=', iCell, &
                             ', k=', k, ' with thickness of ', layerThicknessCur(k, iCell) , '.'
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
   subroutine ocn_prevent_drying(block, dt, rkWeight, err) !{{{

     !-----------------------------------------------------------------
     !
     ! input variables
     !
     !-----------------------------------------------------------------

     type (block_type), intent(in) :: block
     real (kind=RKIND), intent(in) :: dt
     real (kind=RKIND), intent(in) :: rkWeight

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
     real (kind=RKIND), dimension(:,:), pointer :: tend_normalVelocity, layerThickness, layerThicknessEdge


      integer, pointer :: nCells, nEdges
      integer :: iCell, iEdge, k

      type (mpas_pool_type), pointer :: tendPool, meshPool, statePool, scratchPool
      type (mpas_pool_type), pointer :: diagnosticsPool, provisStatePool, forcingPool
      type (mpas_pool_type), pointer :: tracersPool, tracersTendPool, provisTracersPool

      real (kind=RKIND), dimension(:, :), pointer :: normalVelocityCur, normalVelocityProvis, normalVelocityTend
      real (kind=RKIND), dimension(:, :), pointer :: layerThicknessCur, layerThicknessProvis, layerThicknessTend
      real (kind=RKIND), dimension(:, :), pointer :: lowFreqDivergenceCur, lowFreqDivergenceProvis, lowFreqDivergenceTend
      real (kind=RKIND), dimension(:, :), pointer :: normalTransportVelocity, normalGMBolusVelocity

      real (kind=RKIND), dimension(:, :, :), pointer :: tracersGroupCur, tracersGroupProvis, tracersGroupTend

      integer, dimension(:), pointer :: maxLevelCell, maxLevelEdgeTop

      logical, pointer :: config_prescribe_velocity, config_prescribe_thickness, config_use_standardGM
      logical, pointer :: config_use_tracerGroup
      type (mpas_pool_iterator_type) :: groupItr
      character (len=StrKIND) :: modifiedGroupName
      character (len=StrKIND) :: configName

      err = 0
      !call mpas_log_write('Computing provisional variables...')

     call mpas_pool_get_subpool(block % structs, 'diagnostics', diagnosticsPool)
     call mpas_pool_get_subpool(block % structs, 'tend', tendPool)
     call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
     call mpas_pool_get_subpool(block % structs, 'state', statePool)

     call mpas_pool_get_array(diagnosticsPool, 'normalTransportVelocity', normalTransportVelocity)
     call mpas_pool_get_array(diagnosticsPool, 'layerThicknessEdge', layerThicknessEdge)
     call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, 1)
     call mpas_pool_get_array(tendPool, 'normalVelocity', tend_normalVelocity)

      call mpas_pool_get_config(block % configs, 'config_prescribe_velocity', config_prescribe_velocity)
      call mpas_pool_get_config(block % configs, 'config_prescribe_thickness', config_prescribe_thickness)
      call mpas_pool_get_config(block % configs, 'config_use_standardGM', config_use_standardGM)

      call mpas_pool_get_dimension(block % dimensions, 'nCells', nCells)
      call mpas_pool_get_dimension(block % dimensions, 'nEdges', nEdges)

      call mpas_pool_get_subpool(block % structs, 'state', statePool)
      call mpas_pool_get_subpool(block % structs, 'tend', tendPool)
      call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
      call mpas_pool_get_subpool(block % structs, 'scratch', scratchPool)
      call mpas_pool_get_subpool(block % structs, 'diagnostics', diagnosticsPool)
      call mpas_pool_get_subpool(block % structs, 'provis_state', provisStatePool)
      call mpas_pool_get_subpool(block % structs, 'forcing', forcingPool)

      call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
      call mpas_pool_get_subpool(tendPool, 'tracersTend', tracersTendPool)

      call mpas_pool_get_subpool(provisStatePool, 'tracers', provisTracersPool)

      call mpas_pool_get_array(statePool, 'normalVelocity', normalVelocityCur, 1)
      call mpas_pool_get_array(statePool, 'layerThickness', layerThicknessCur, 1)
      call mpas_pool_get_array(statePool, 'lowFreqDivergence', lowFreqDivergenceCur, 1)

      call mpas_pool_get_array(provisStatePool, 'normalVelocity', normalVelocityProvis, 1)
      call mpas_pool_get_array(provisStatePool, 'layerThickness', layerThicknessProvis, 1)
      call mpas_pool_get_array(provisStatePool, 'lowFreqDivergence', lowFreqDivergenceProvis, 1)

      call mpas_pool_get_array(tendPool, 'normalVelocity', normalVelocityTend)
      call mpas_pool_get_array(tendPool, 'layerThickness', layerThicknessTend)
      call mpas_pool_get_array(tendPool, 'lowFreqDivergence', lowFreqDivergenceTend)

      call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
      call mpas_pool_get_array(meshPool, 'maxLevelEdgeTop', maxLevelEdgeTop)

      call mpas_pool_get_array(diagnosticsPool, 'normalTransportVelocity', normalTransportVelocity)
      call mpas_pool_get_array(diagnosticsPool, 'normalGMBolusVelocity', normalGMBolusVelocity)

     ! get provisional values prior to updating tendency to prevent drying
     ! (provisional values use estimated tendency)


      call mpas_threading_barrier()

      !$omp do schedule(runtime) private(k)
      do iEdge = 1, nEdges
         do k = 1, maxLevelEdgeTop(iEdge)
            normalVelocityProvis(k, iEdge) = normalVelocityCur(k, iEdge) + rkWeight &
                                           * normalVelocityTend(k, iEdge)
         end do
      end do
      !$omp end do


      !$omp do schedule(runtime) private(k)
      do iCell = 1, nCells
         do k = 1, maxLevelCell(iCell)
            layerThicknessProvis(k, iCell) = layerThicknessCur(k, iCell) + rkWeight &
                                           * layerThicknessTend(k, iCell)
         end do
      end do
      !$omp end do

      call mpas_pool_begin_iteration(tracersPool)
      do while ( mpas_pool_get_next_member(tracersPool, groupItr) )
         if ( groupItr % memberType == MPAS_POOL_FIELD ) then
            configName = 'config_use_' // trim(groupItr % memberName)
            call mpas_pool_get_config(block % configs, configName, config_use_tracerGroup)

            if ( config_use_tracerGroup ) then
               call mpas_pool_get_array(tracersPool, groupItr % memberName, tracersGroupCur, 1)
               call mpas_pool_get_array(provisTracersPool, groupItr % memberName, tracersGroupProvis, 1)

               modifiedGroupName = trim(groupItr % memberName) // 'Tend'
               call mpas_pool_get_array(tracersTendPool, modifiedGroupName, tracersGroupTend)
               if ( associated(tracersGroupProvis) .and. associated(tracersGroupCur) .and. associated(tracersGroupTend) ) then
                  !$omp do schedule(runtime) private(k)
                  do iCell = 1, nCells
                     do k = 1, maxLevelCell(iCell)
                        tracersGroupProvis(:, k, iCell) = ( layerThicknessCur(k, iCell) * tracersGroupCur(:, k, iCell)  &
                                                 + rkWeight * tracersGroupTend(:, k, iCell) &
                                                   ) / layerThicknessProvis(k, iCell)
                     end do

                  end do
                  !$omp end do
               end if
            end if
         end if
      end do

      if (associated(lowFreqDivergenceCur)) then
         !$omp do schedule(runtime)
         do iCell = 1, nCells
            lowFreqDivergenceProvis(:, iCell) = lowFreqDivergenceCur(:, iCell) + rkWeight &
                                              * lowFreqDivergenceTend(:, iCell)
         end do
         !$omp end do
      end if

      if (config_prescribe_velocity) then
         !$omp do schedule(runtime)
         do iEdge = 1, nEdges
            normalVelocityProvis(:, iEdge) = normalVelocityCur(:, iEdge)
         end do
         !$omp end do
      end if

      if (config_prescribe_thickness) then
         !$omp do schedule(runtime)
         do iCell = 1, nCells
            layerThicknessProvis(:, iCell) = layerThicknessCur(:, iCell)
         end do
         !$omp end do
      end if
      call mpas_threading_barrier()

      call ocn_diagnostic_solve(dt, provisStatePool, forcingPool, meshPool, diagnosticsPool, scratchPool, tracersPool, 1)
      call mpas_threading_barrier()

      ! ------------------------------------------------------------------
      ! Accumulating various parametrizations of the transport velocity
      ! ------------------------------------------------------------------
      !$omp do schedule(runtime)
      do iEdge = 1, nEdges
         normalTransportVelocity(:, iEdge) = normalVelocityProvis(:, iEdge)
      end do
      !$omp end do
      call mpas_threading_barrier()

      ! Compute normalGMBolusVelocity, relativeSlope and RediDiffVertCoef if respective flags are turned on
      if (config_use_standardGM) then
         call ocn_gm_compute_Bolus_velocity(diagnosticsPool, meshPool, scratchPool)
      end if
      call mpas_threading_barrier()

      if (config_use_standardGM) then
         !$omp do schedule(runtime)
         do iEdge = 1, nEdges
            normalTransportVelocity(:, iEdge) = normalTransportVelocity(:, iEdge) + normalGMBolusVelocity(:,iEdge)
         end do
         !$omp end do
      end if
      call mpas_threading_barrier()
      ! ------------------------------------------------------------------
      ! End: Accumulating various parametrizations of the transport velocity
      ! ------------------------------------------------------------------
      !call mpas_log_write('Done with provisional variabiles')

     ! ensure cells stay wet by selectively damping cells with a damping tendency to make sure tendency doesn't dry cells
     call ocn_wetting_drying_tend(meshPool, layerThicknessEdge, layerThicknessProvis, normalTransportVelocity, rkWeight, normalVelocityTend, err)

   end subroutine ocn_prevent_drying !}}}

!***********************************************************************
!
!  routine ocn_wetting_drying_tend
!
!> \brief   Computes tendency to prevent cell drying
!> \author  Phillip J. Wolfram
!> \date    03/19/2018
!> \details
!>  This routine adds edge-damping to prevent cells from drying.
!
!-----------------------------------------------------------------------
   subroutine ocn_wetting_drying_tend(meshPool, layerThicknessEdge, layerThickness, &
       normalVelocity, dt, tend, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      type (mpas_pool_type), intent(in) :: &
         meshPool           !< Input: horizonal mesh information

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         layerThickness    !< Input: layer thickness at old time

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
         tend          !< Input/Output: velocity tendency

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

      real (kind=RKIND) :: divOutFlux, invAreaCell
      real (kind=RKIND), dimension(:), pointer :: dvEdge, areaCell

      real (kind=RKIND), pointer :: config_drying_min_cell_height
      real(kind=RKIND), parameter :: eps = 1.0e-12_RKIND

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
      nCells = nCellsArray( 2 )

      call mpas_threading_barrier()

      call mpas_log_write('IN WETTING AND DRYING TEND $i', intArgs=(/nCells/))

      ! need predicted transport velocity to limit tendency

      !! !$omp do schedule(runtime) private(invAreaCell, i, iEdge, k)
      do iCell = 1, nCells
        invAreaCell = 1.0_RKIND / areaCell(iCell)
        do k = 1, maxLevelCell(iCell)
          divOutFlux = 0.0_RKIND
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

          ! if layer thickness is too small, limit divergence flux outwards with tendency
          if ((layerThickness(k, iCell) - dt*divOutFlux ) < config_drying_min_cell_height) then
            !call mpas_log_write('LIMIT FLUX')
            ! limit divOutFlux out of cell to keep it wet
            divOutFlux = (layerThickness(k, iCell) - config_drying_min_cell_height) / (dt*divOutFlux)

            do i = 1, nEdgesOnCell(iCell)
              iEdge = edgesOnCell(i, iCell)
              if (k <= maxLevelEdgeBot(iEdge)) then
                if ( normalVelocity(k, iEdge) * edgeSignOnCell(i, iCell) < 0.0_RKIND ) then
                  ! each outgoing velocity is penalized (but not the incoming, wetting velocities)
                  tend(k, iEdge) = tend(k, iEdge) - (1.0_RKIND - divOutFlux)/dt * normalVelocity(k, iEdge)
                end if
              end if
            end do
          end if

        end do
      end do
      !! !$omp end do

      call mpas_threading_barrier()
      call mpas_log_write('OUT OF WETTING AND DRYING TEND')

   end subroutine ocn_wetting_drying_tend !}}}

end module ocn_wetting_drying

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

! vim: foldmethod=marker
