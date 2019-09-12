! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_time_integration_rk4
!
!> \brief MPAS ocean RK4 Time integration scheme
!> \author Mark Petersen, Doug Jacobsen, Todd Ringler
!> \date   September 2011
!> \details
!>  This module contains the RK4 time integration routine.
!
!-----------------------------------------------------------------------

module ocn_time_integration_rk4

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use mpas_dmpar
   use mpas_threading
   use mpas_vector_reconstruction
   use mpas_spline_interpolation
   use mpas_timer

   use ocn_constants
   use ocn_tendency
   use ocn_diagnostics
   use ocn_gm

   use ocn_equation_of_state
   use ocn_vmix
   use ocn_time_average_coupled
   use ocn_wetting_drying

   use ocn_effective_density_in_land_ice

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

   public :: ocn_time_integrator_rk4

   contains

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_time_integrator_rk4
!
!> \brief MPAS ocean RK4 Time integration scheme
!> \author Mark Petersen, Doug Jacobsen, Todd Ringler
!> \date   September 2011
!> \details
!>  This routine integrates one timestep (dt) using an RK4 time integrator.
!
!-----------------------------------------------------------------------

   subroutine ocn_time_integrator_rk4(domain, dt)!{{{
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   ! Advance model state forward in time by the specified time step using
   !   4th order Runge-Kutta
   !
   ! Input: domain - current model state in time level 1 (e.g., time_levs(1)state%h(:,:))
   !                 plus mesh meta-data
   ! Output: domain - upon exit, time level 2 (e.g., time_levs(2)%state%h(:,:)) contains
   !                  model state advanced forward in time by dt seconds
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      implicit none

      type (domain_type), intent(inout) :: domain !< Input/Output: domain information
      real (kind=RKIND), intent(in) :: dt !< Input: timestep

      integer :: iCell, iEdge, k, i, err
      type (block_type), pointer :: block

      type (mpas_pool_type), pointer :: tendPool
      type (mpas_pool_type), pointer :: tracersTendPool
      type (mpas_pool_type), pointer :: statePool
      type (mpas_pool_type), pointer :: tracersPool
      type (mpas_pool_type), pointer :: meshPool
      type (mpas_pool_type), pointer :: provisStatePool
      type (mpas_pool_type), pointer :: provisTracersPool
      type (mpas_pool_type), pointer :: diagnosticsPool
      type (mpas_pool_type), pointer :: verticalMeshPool
      type (mpas_pool_type), pointer :: forcingPool
      type (mpas_pool_type), pointer :: scratchPool
      type (mpas_pool_type), pointer :: swForcingPool

      integer :: rk_step

      type (mpas_pool_type), pointer :: nextProvisPool, prevProvisPool

      real (kind=RKIND), dimension(4) :: rk_weights, rk_substep_weights

      real (kind=RKIND) :: coef
      real (kind=RKIND), dimension(:,:), pointer :: &
        vertViscTopOfEdge, vertDiffTopOfCell

      ! Dimensions
      integer, pointer :: nCells, nEdges, nVertLevels, num_tracers

      ! Config options
      logical, pointer :: config_prescribe_velocity, config_prescribe_thickness
      logical, pointer :: config_filter_btr_mode, config_use_freq_filtered_thickness
      logical, pointer :: config_use_standardGM
      logical, pointer :: config_use_cvmix_kpp
      logical, pointer :: config_use_tracerGroup
      logical, pointer :: config_disable_thick_all_tend
      logical, pointer :: config_disable_vel_all_tend
      logical, pointer :: config_disable_tr_all_tend
      real (kind=RKIND), pointer :: config_mom_del4
      real (kind=RKIND), pointer :: config_drying_min_cell_height
      logical, pointer :: config_use_wetting_drying
      logical, pointer :: config_verify_not_dry
      logical, pointer :: config_prevent_drying
      logical, pointer :: config_zero_drying_velocity
      character (len=StrKIND), pointer :: config_land_ice_flux_mode

      ! State indices
      integer, pointer :: indexTemperature
      integer, pointer :: indexSalinity

      ! Diagnostics Indices
      integer, pointer :: indexSurfaceVelocityZonal, indexSurfaceVelocityMeridional
      integer, pointer :: indexSSHGradientZonal, indexSSHGradientMeridional

      ! Mesh array pointers
      integer, dimension(:), pointer :: maxLevelCell, maxLevelEdgeTop
      real (kind=RKIND), dimension(:), pointer :: bottomDepth

      ! Provis Array Pointers
      real (kind=RKIND), dimension(:,:), pointer :: normalVelocityProvis, layerThicknessProvis
      real (kind=RKIND), dimension(:,:), pointer :: highFreqThicknessProvis
      real (kind=RKIND), dimension(:,:), pointer :: lowFreqDivergenceProvis
      real (kind=RKIND), dimension(:,:,:), pointer :: tracersGroupProvis

      ! Tend Array Pointers
      real (kind=RKIND), dimension(:,:), pointer :: highFreqThicknessTend, lowFreqDivergenceTend, normalVelocityTend, &
                                                    layerThicknessTend
      real (kind=RKIND), dimension(:,:,:), pointer :: tracersGroupTend

      ! Diagnostics Array Pointers
      real (kind=RKIND), dimension(:,:), pointer :: layerThicknessEdge
      real (kind=RKIND), dimension(:,:), pointer :: vertAleTransportTop
      real (kind=RKIND), dimension(:,:), pointer :: normalTransportVelocity, normalGMBolusVelocity
      real (kind=RKIND), dimension(:,:), pointer :: velocityX, velocityY, velocityZ
      real (kind=RKIND), dimension(:,:), pointer :: velocityZonal, velocityMeridional
      real (kind=RKIND), dimension(:), pointer :: gradSSH
      real (kind=RKIND), dimension(:), pointer :: gradSSHX, gradSSHY, gradSSHZ
      real (kind=RKIND), dimension(:), pointer :: gradSSHZonal, gradSSHMeridional
      real (kind=RKIND), dimension(:,:), pointer :: surfaceVelocity, sshGradient

      ! State Array Pointers
      real (kind=RKIND), dimension(:,:), pointer :: normalVelocityCur, normalVelocityNew
      real (kind=RKIND), dimension(:,:), pointer :: layerThicknessCur, layerThicknessNew
      real (kind=RKIND), dimension(:,:), pointer :: highFreqThicknessCur, highFreqThicknessNew
      real (kind=RKIND), dimension(:,:), pointer :: lowFreqDivergenceCur, lowFreqDivergenceNew
      real (kind=RKIND), dimension(:), pointer :: sshCur, sshNew

      real (kind=RKIND), dimension(:,:,:), pointer :: tracerGroup, tracersCur, tracersNew

      ! Diagnostics Field Pointers
      type (field1DReal), pointer :: boundaryLayerDepthField, effectiveDensityField
      type (field2DReal), pointer :: normalizedRelativeVorticityEdgeField, divergenceField, relativeVorticityField

      ! State/Tend Field Pointers
      type (field2DReal), pointer :: normalVelocityField, layerThicknessField
      type (field2DReal), pointer :: wettingVelocityField
      type (field3DReal), pointer :: tracersGroupField

      ! Tracer Group Iteartion
      type (mpas_pool_iterator_type) :: groupItr
      character (len=StrKIND) :: modifiedGroupName
      character (len=StrKIND) :: configName

      ! Tidal boundary condition
      logical, pointer :: config_use_tidal_forcing
      character (len=StrKIND), pointer :: config_tidal_forcing_type
      real (kind=RKIND), dimension(:), pointer :: tidalInputMask, tidalBCValue
      real (kind=RKIND), dimension(:,:), pointer :: restingThickness
      real (kind=RKIND) :: totalDepth

      ! Get config options
      call mpas_pool_get_config(domain % configs, 'config_mom_del4', config_mom_del4)
      call mpas_pool_get_config(domain % configs, 'config_filter_btr_mode', config_filter_btr_mode)
      call mpas_pool_get_config(domain % configs, 'config_prescribe_velocity', config_prescribe_velocity)
      call mpas_pool_get_config(domain % configs, 'config_prescribe_thickness', config_prescribe_thickness)
      call mpas_pool_get_config(domain % configs, 'config_use_freq_filtered_thickness', config_use_freq_filtered_thickness)
      call mpas_pool_get_config(domain % configs, 'config_use_standardGM', config_use_standardGM)
      call mpas_pool_get_config(domain % configs, 'config_use_cvmix_kpp', config_use_cvmix_kpp)
      call mpas_pool_get_config(domain % configs, 'config_land_ice_flux_mode', config_land_ice_flux_mode)
      call mpas_pool_get_config(domain % configs, 'config_disable_vel_all_tend', config_disable_vel_all_tend)
      call mpas_pool_get_config(domain % configs, 'config_disable_thick_all_tend', config_disable_thick_all_tend)
      call mpas_pool_get_config(domain % configs, 'config_disable_tr_all_tend', config_disable_tr_all_tend)
      call mpas_pool_get_config(domain % configs, 'config_use_wetting_drying', config_use_wetting_drying)
      call mpas_pool_get_config(domain % configs, 'config_prevent_drying', config_prevent_drying)
      call mpas_pool_get_config(domain % configs, 'config_verify_not_dry', config_verify_not_dry)
      call mpas_pool_get_config(domain % configs, 'config_drying_min_cell_height', config_drying_min_cell_height)
      call mpas_pool_get_config(domain % configs, 'config_zero_drying_velocity', config_zero_drying_velocity)
      call mpas_pool_get_config(domain % configs, 'config_use_tidal_forcing', config_use_tidal_forcing)
      call mpas_pool_get_config(domain % configs, 'config_tidal_forcing_type', config_tidal_forcing_type)

      !
      ! Initialize time_levs(2) with state at current time
      ! Initialize first RK state
      ! Couple tracers time_levs(2) with layerThickness in time-levels
      ! Initialize RK weights
      !
      block => domain % blocklist
      do while (associated(block))
         call mpas_pool_get_subpool(block % structs, 'state', statePool)
         call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)

         call mpas_pool_create_pool(provisStatePool)
         call mpas_pool_clone_pool(statePool, provisStatePool, 1)
         call mpas_pool_destroy_pool(provisStatePool)

         call mpas_threading_barrier()

         block => block % next
      end do

      call mpas_threading_barrier()

   end subroutine ocn_time_integrator_rk4!}}}


end module ocn_time_integration_rk4

! vim: foldmethod=marker
