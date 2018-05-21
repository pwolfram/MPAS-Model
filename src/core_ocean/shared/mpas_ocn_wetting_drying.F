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

   public :: ocn_wetting_drying_verify, ocn_prevent_drying_rk4, &
     ocn_prevent_drying_split_explicit

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

      type (mpas_pool_type), pointer :: statePool, meshPool, tendPool
      integer, dimension(:), pointer :: maxLevelCell
      real (kind=RKIND), dimension(:), pointer :: sshSubcycleNew, bottomDepth
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
      call mpas_pool_get_array(statePool, 'sshSubcycle', sshSubcycleNew, 2)
      call mpas_pool_get_array(meshPool, 'bottomDepth', bottomDepth)

      err = 0

      call mpas_log_write( 'Verifying that cells are not dry...')

      ! check to make sure that there is no layer that is too dry
      minThickness = +1.0E34
      do iCell = 1, nCellsSolve
        do k = 1, maxLevelCell(iCell)
          !call mpas_log_write( 'Layer thickness = $r', realArgs=(/ layerThick/))
          ! use ssh as a proxy too for baroclinic mode
          layerThick = min(layerThicknessNew(k, iCell), (sshSubcycleNew(iCell)+bottomDepth(iCell))/maxLevelCell(iCell))
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
!  routine ocn_prevent_drying_split_explicit
!
!> \brief   Prevents velocity tendency from causing cell drying
!> \author  Phillip J. Wolfram
!> \date    05/01/2018
!> \details
!>  This routine modifies velocity tendency to prevent cell drying.
!
!-----------------------------------------------------------------------
   subroutine ocn_prevent_drying_split_explicit(meshPool, barotropicWettingVelocity, &
       dt, nBtrSubcycles, normalBarotropicVelocitySubcycleCur, normalBarotropicVelocitySubcycleNew, &
       sshSubcycleCur, config_btr_gam1_velWt1, config_zero_drying_velocity, err) !{{{

     !-----------------------------------------------------------------
     !
     ! input variables
     !
     !-----------------------------------------------------------------

     type (mpas_pool_type), pointer, intent(in) :: meshPool
     real (kind=RKIND), intent(in) :: dt
     integer, intent(in) :: nBtrSubcycles
     real (kind=RKIND), dimension(:), intent(in) :: normalBarotropicVelocitySubcycleCur, &
                                                    normalBarotropicVelocitySubcycleNew
     real (kind=RKIND), intent(in) :: config_btr_gam1_velWt1
     real (kind=RKIND), dimension(:), intent(in) :: sshSubcycleCur
     logical, intent(in) :: config_zero_drying_velocity

     !-----------------------------------------------------------------
     !
     ! input/output variables
     !
     !-----------------------------------------------------------------
     real (kind=RKIND), dimension(:), intent(inout) :: barotropicWettingVelocity

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
     integer, dimension(:,:), pointer :: edgesOnCell, cellsOnEdge, edgeSignOnCell
     real (kind=RKIND), dimension(:), pointer :: dvEdgea, areaCell, dvEdge, bottomDepth
     integer, dimension(:), pointer :: nCellsArray, nEdgesOnCell, maxLevelCell
     real (kind=RKIND), pointer :: config_drying_min_cell_height

     real (kind=RKIND) :: invAreaCell, divOutFlux, edgeVel, sshEdge, layerThickness, thicknessSum
     integer :: nCells

     integer :: i, iEdge, iCell, cell1, cell2

     err = 0

     call mpas_pool_get_config(ocnConfigs, 'config_drying_min_cell_height', config_drying_min_cell_height)

     call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)
     call mpas_pool_get_array(meshPool, 'edgesOnCell', edgesOnCell)
     call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
     call mpas_pool_get_array(meshPool, 'edgeSignOnCell', edgeSignOnCell)
     call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
     call mpas_pool_get_array(meshPool, 'bottomDepth', bottomDepth)
     call mpas_pool_get_array(meshPool, 'dvEdge', dvEdge)
     call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
     call mpas_pool_get_dimension(meshPool, 'nCellsArray', nCellsArray)

     ! Only need to compute over 0 and 1 halos
     nCells = nCellsArray( 1 )

     call mpas_threading_barrier()

     call mpas_log_write('IN WETTING AND DRYING BAROTROPIC $i', intArgs=(/nCells/))

     ! need predicted transport velocity to limit drying flux
     ! !$omp do schedule(runtime) private(invAreaCell, i, iEdge, k)
     do iCell = 1, nCells
       invAreaCell = 1.0_RKIND / areaCell(iCell)
       ! can switch with maxLevelEdgeBot(iEdge)
       divOutFlux = 0.0_RKIND
       !layerThickness = layerThicknessCur(k, iCell)
       !layerThickness = layerThicknessProvis(k, iCell)
       !layerThickness = min(layerThicknessProvis(k, iCell), layerThicknessCur(k, iCell))
       layerThickness = sshSubcycleCur(iCell) + bottomDepth(iCell)
       !call mpas_log_write('prov = $r, curr=$r', realArgs=(/ layerThicknessProvis(k,iCell), layerThicknessCur(k,iCell)/))
       !layerThickness = layerThicknessCur(k, iCell)
       do i = 1, nEdgesOnCell(iCell)
         iEdge = edgesOnCell(i, iCell)

         cell1 = cellsOnEdge(1, iEdge)
         cell2 = cellsOnEdge(2, iEdge)

         sshEdge = 0.5_RKIND * (sshSubcycleCur(cell1) + sshSubcycleCur(cell2) )

         ! method 0: orig, works only without pbc:
         !thicknessSum = sshEdge + refBottomDepthTopOfCell(maxLevelEdgeTop(iEdge)+1)

         ! method 1, matches method 0 without pbcs, works with pbcs.
         thicknessSum = sshEdge + min(bottomDepth(cell1), bottomDepth(cell2))

         ! method 2: may be better than method 1.
         ! Take average  of full thickness at two neighboring cells.
         !thicknessSum = sshEdge + 0.5 *( bottomDepth(cell1) + bottomDepth(cell2) )

         ! normal edge velocity
         edgeVel = (1.0_RKIND-config_btr_gam1_velWt1) * normalBarotropicVelocitySubcycleCur(iEdge) &
           + config_btr_gam1_velWt1 * normalBarotropicVelocitySubcycleNew(iEdge)

         ! only consider divergence flux leaving the cell
         !call mpas_log_write('u = $r, sign=$i', realArgs=(/ normalVelocity(k,iEdge)/), intArgs=(/edgesSignOnCell(i,iCell)/))
         if ( edgeVel * edgeSignOnCell(i, iCell) < 0.0_RKIND ) then
           !call mpas_log_write('LIMIT FLUX1')
           ! potential problem here, if we have large flux out of one face but small out of the other, small flux could get
           ! lost due to truncation error
           divOutFlux = divOutFlux - edgeVel * edgeSignOnCell(i, iCell) &
             * thicknessSum * dvEdge(iEdge)  * invAreaCell
         end if
       end do
       !call mpas_log_write('divOutFlux= $r, layerThickness = $r, dt= $r, edgevel= $r', realArgs=(/divOutFlux, layerThickness, dt, edgeVel/))

       divOutFlux = abs(divOutFlux)
       ! if layer thickness is too small, limit divergence flux outwards with opposite velocity
       if ((layerThickness - (divOutFlux*dt)/nBtrSubcycles) <= (config_drying_min_cell_height*maxLevelCell(iCell)))  then
         call mpas_log_write('LIMIT FLUX for h = $r div = $r dt = $r cellheight = $r ', realArgs=(/layerThickness, divOutFlux, dt, config_drying_min_cell_height/))
         call mpas_log_write('LIMIT FLUX for cell= $i n= $i maxlevel=$i', intArgs=(/iCell, nBtrSubcycles, maxLevelCell(iCell)/))
         ! limit divOutFlux out of cell to keep it wet
         divOutFlux = (layerThickness - (config_drying_min_cell_height*maxLevelCell(iCell) + eps)) / ((divOutFlux*dt)/nBtrSubcycles + eps)

         do i = 1, nEdgesOnCell(iCell)
           iEdge = edgesOnCell(i, iCell)

           ! normal edge velocity
           edgeVel = (1.0_RKIND-config_btr_gam1_velWt1) * normalBarotropicVelocitySubcycleCur(iEdge) &
             + config_btr_gam1_velWt1 * normalBarotropicVelocitySubcycleNew(iEdge)

           if ( edgeVel * edgeSignOnCell(i, iCell) < 0.0_RKIND ) then
             ! each outgoing velocity is penalized (but not the incoming, wetting velocities)
             !call mpas_log_write('theta = $r, $r', realArgs=(/ divOutFlux, min(max(0.0_RKIND, 1.0_RKIND - divOutFlux), 1.0_RKIND) /))
             ! square the fractional term to make values near zero go to zero much quicker (to prevent threshold from being hit)
             !barotropicWettingVelocity(iEdge) =  - (min(max(0.0_RKIND, 1.0_RKIND - (divOutFlux*divOutFlux)), 1.0_RKIND)) * edgeVel
             ! just completely remove the velocity
             barotropicWettingVelocity(iEdge) =  - edgeVel
           end if
         end do
       end if

     end do
     !! !$omp end do

     call mpas_threading_barrier()
     call mpas_log_write('OUT OF WETTING AND DRYING BAROTROPIC')

   end subroutine ocn_prevent_drying_split_explicit !}}}

!***********************************************************************
!
!  routine ocn_prevent_drying_rk4
!
!> \brief   Prevents velocity tendency from causing cell drying
!> \author  Phillip J. Wolfram
!> \date    03/20/2018
!> \details
!>  This routine modifies velocity tendency to prevent cell drying.
!
!-----------------------------------------------------------------------
   subroutine ocn_prevent_drying_rk4(block, dt, rkWeight, config_zero_drying_velocity, err) !{{{

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

   end subroutine ocn_prevent_drying_rk4 !}}}

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

      !call mpas_log_write('IN WETTING AND DRYING LAYER $i', intArgs=(/nCells/))

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
      !call mpas_log_write('OUT OF WETTING AND DRYING LAYER')

   end subroutine ocn_wetting_drying_wettingVelocity !}}}

end module ocn_wetting_drying

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

! vim: foldmethod=marker
