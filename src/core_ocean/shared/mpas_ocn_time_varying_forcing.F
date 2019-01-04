!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_time_varying_forcing
!
!> \brief Module to handle time-varying forcing for the ocean core
!> \author Steven Brus, LANL
!> \date 2013-2014
!> \details
!>  Manages the wind and pressure forcing for the ocean.
!>  Based very closely on the seaice forcing module and
!>  framework written by Adrian Turner.
!
!-----------------------------------------------------------------------

module ocn_time_varying_forcing

  use mpas_derived_types
  use mpas_pool_routines
  use mpas_timekeeping
  use mpas_forcing
  use mpas_stream_manager
  use ocn_framework_forcing
  use mpas_log, only: mpas_log_write

  implicit none

  private
  public :: &
       ocn_time_varying_forcing_init, &
       ocn_time_varying_forcing_get, &
       ocn_time_varying_forcing_write_restart_times

contains

!-----------------------------------------------------------------------
! initialization
!-----------------------------------------------------------------------

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_time_varying_forcing_init
!
!> \brief Driver for ocean core forcing initialization
!> \author Steven Brus, LANL
!> \date
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine ocn_time_varying_forcing_init(domain)

    type (domain_type) :: domain

    logical, pointer :: &
         config_use_atmospheric_forcing

    call MPAS_pool_get_config(domain % configs, "config_use_time_varying_atmospheric_forcing", config_use_atmospheric_forcing)

    ! init the atmospheric forcing
    if (config_use_atmospheric_forcing) then
       call init_atmospheric_forcing(domain)
    endif

    ! init other forcings, to be added later as needed
    !if (config_use_time_varying_****_forcing) then
    !  call init_****_forcing(domain)
    !endif

  end subroutine ocn_time_varying_forcing_init

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  init_atmospheric_forcing
!
!> \brief
!> \author Steven Brus, LANL
!> \date
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine init_atmospheric_forcing(domain)

    type (domain_type) :: domain

    character(len=strKIND), pointer :: &
         config_atmospheric_forcing_type

    call MPAS_pool_get_config(domain % configs, "config_time_varying_atmospheric_forcing_type", config_atmospheric_forcing_type)

    select case (trim(config_atmospheric_forcing_type))
      case ("WINDPRES")
         call init_atmospheric_forcing_WINDPRES(domain)
      case default
         call mpas_log_write("Atmospheric forcing type unknown: "//trim(config_atmospheric_forcing_type), MPAS_LOG_CRIT)
    end select

  end subroutine init_atmospheric_forcing

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  init_atmospheric_forcing_WINDPRES
!
!> \brief Initialize the forcing objects for wind and pressure
!> \author Steven Brus, LANL
!> \date 2013-2014
!> \details
!>  This routine calls the MPAS_forcing module subroutines that initializes
!>  the forcings type
!
!-----------------------------------------------------------------------

  subroutine init_atmospheric_forcing_WINDPRES(domain)

    type(domain_type) :: domain

    character(len=strKIND), pointer :: &
         config_forcing_start_time, &
         config_forcing_cycle_start, &
         config_forcing_cycle_duration

    logical, pointer :: &
         config_do_restart

    character(len=strKIND) :: &
         forcingInterval, &
         forcingReferenceTime

    ! get atmospheric forcing configuration options
    call MPAS_pool_get_config(&
         domain % configs, &
         "config_time_varying_atmospheric_forcing_start_time", &
         config_forcing_start_time)

    call MPAS_pool_get_config(&
         domain % configs, &
         "config_time_varying_atmospheric_forcing_cycle_start", &
         config_forcing_cycle_start)

    call MPAS_pool_get_config(&
         domain % configs, &
         "config_time_varying_atmospheric_forcing_cycle_duration", &
         config_forcing_cycle_duration)

    call MPAS_pool_get_config(&
         domain % configs, &
         "config_do_restart", &
         config_do_restart)

    ! create the six hourly forcing group
    call MPAS_forcing_init_group(&
         forcingGroupHead, &                     ! forcingGroupHead
         "ocn_atmospheric_forcing", &            ! forcingGroupName
         domain, &                               ! domain
         config_forcing_start_time, &            ! startTimeStr
         config_forcing_cycle_start, &           ! forcingCycleStart
         config_forcing_cycle_duration, &        ! forcingCycleDuration
         config_do_restart)!, &                  ! restart
         !.false.)                               ! forcingCycleStartInclusive

    forcingInterval = "01:00:00"
    forcingReferenceTime = "0001-01-01_00:00:00"


    ! windSpeedU init
    call MPAS_forcing_init_field(&
         domain % streamManager, &               ! streamManager
         forcingGroupHead, &                     ! forcingGroupHead
         "ocn_atmospheric_forcing", &            ! forcingGroupName
         "windSpeedU", &                         ! forcingName
         "atmospheric_forcing", &                ! forcingStreamID
         "timeVaryingForcing", &                 ! poolname
         "windSpeedU", &                         ! fieldname
         "linear", &                             ! interpolationType
         forcingReferenceTime, &                 ! forcingReferenceTimeStr
         forcingInterval)!,                      ! forcingIntervalStr
         !"next")                                ! forcingInitializationType

    ! windSpeedV init
    call MPAS_forcing_init_field(&
         domain % streamManager, &
         forcingGroupHead, &
         "ocn_atmospheric_forcing", &
         "windSpeedV", &
         "atmospheric_forcing", &
         "timeVaryingForcing", &
         "windSpeedV", &
         "linear", &
         forcingReferenceTime, &
         forcingInterval)!, &
         !"next")

    ! atmosphericPressure init
    call MPAS_forcing_init_field(&
         domain % streamManager, &
         forcingGroupHead, &
         "ocn_atmospheric_forcing", &
         "atmosPressure", &
         "atmospheric_forcing", &
         "timeVaryingForcing", &
         "atmosPressure", &
         "linear", &
         forcingReferenceTime, &
         forcingInterval)!, &
         !"next")

    call MPAS_forcing_init_field_data(&
         forcingGroupHead, &                     ! forcingGroupHead
         "ocn_atmospheric_forcing", &            ! forcingGroupName
         domain % streamManager, &               ! streamManager
         config_do_restart, &                    ! restart
         .false.)                                ! interpolateAtInit

  end subroutine init_atmospheric_forcing_WINDPRES

!-----------------------------------------------------------------------
! runtime
!-----------------------------------------------------------------------

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_time_varying_forcing_get
!
!> \brief Retrieve forcing data during time stepping
!> \author Steven Brus, LANL
!> \date 2013-2014
!> \details
!>  This routine calls the MPAS_forcing routine that will perform the
!>  forcing data aquisition and interpolation during timestepping
!
!-----------------------------------------------------------------------

  subroutine ocn_time_varying_forcing_get(&
       streamManager, &
       domain, &
       simulationClock)

    type (MPAS_streamManager_type), intent(inout) :: streamManager

    type (domain_type) :: domain

    type (MPAS_clock_type) :: simulationClock

    logical, pointer :: &
         config_use_atmospheric_forcing

    call MPAS_pool_get_config(domain % configs, "config_use_time_varying_atmospheric_forcing", config_use_atmospheric_forcing)

    if (config_use_atmospheric_forcing) then

        call atmospheric_forcing(&
             streamManager, &
             domain, &
             simulationClock)

    endif

    ! call other forcings, to be added if needed
    ! if (config_use_****_forcing) then
    !   call ****_forcing(&
    !        streamManager, &
    !        domain, &
    !        simulationClock)
    ! endif


  end subroutine ocn_time_varying_forcing_get

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  atmospheric_forcing
!
!> \brief
!> \author Steven Brus, LANL
!> \date
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine atmospheric_forcing(&
       streamManager, &
       domain, &
       simulationClock)

    type (MPAS_streamManager_type), intent(inout) :: streamManager

    type (domain_type) :: domain

    type (MPAS_clock_type) :: simulationClock

    type (MPAS_time_type) :: currentForcingTime

    type (block_type), pointer :: block

    character(len=StrKIND), pointer :: config_dt

    character(len=strKIND), pointer :: config_atmospheric_forcing_type

    type (MPAS_timeInterval_type) :: timeStepESMF

    real (kind=RKIND) :: dtSim

    integer :: err

    ! configurations
    call mpas_pool_get_config(domain % configs, 'config_dt', config_dt)
    call mpas_pool_get_config(domain % configs, 'config_time_varying_atmospheric_forcing_type', config_atmospheric_forcing_type)

    ! convert config_dt to real
    call mpas_set_timeInterval(timeStepESMF, timeString=config_dt,ierr=err)
    call mpas_get_timeInterval(timeStepESMF, dt=dtSim)

    ! use the forcing layer to get data
    call MPAS_forcing_get_forcing(&
         forcingGroupHead, &               ! forcingGroupHead
         "ocn_atmospheric_forcing", &      ! forcingGroupName
         streamManager, &                  ! streamManager
         dtSim)                            ! dt

    call MPAS_forcing_get_forcing_time(&
         forcingGroupHead, &               ! forcingGroupHead
         "ocn_atmospheric_forcing", &      ! forcingGroupName
         currentForcingTime)               ! forcingTime


    ! perform post forcing
    block => domain % blocklist
    do while (associated(block))

       ! apply drag law to compute wind stress from wind speed
       if (trim(config_atmospheric_forcing_type) == "WINDPRES") then
          call post_atmospheric_forcing(block)
       endif

       block => block % next
    end do

  end subroutine atmospheric_forcing

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  post_atmospheric_forcing
!
!> \brief
!> \author Steven Brus, LANL
!> \date
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine post_atmospheric_forcing(block)

    type (block_type), pointer :: block

    type (MPAS_time_type) :: currentForcingTime

    type (mpas_pool_type), pointer :: &
         mesh, &
         forcingPool, &
         timeVaryingForcingPool, &
         diagnosticsPool

    real(kind=RKIND), dimension(:), pointer :: &
         windSpeedU, &
         windSpeedV, &
         windSpeedMagnitude, &
         windStressZonal, &
         windStressMeridional, &
         atmosPressure, &
         atmosphericPressure

    real(kind=RKIND) :: &
         windStressCoefficient, &
         rhoAir, &
         ramp, &
         currentTime

    real(kind=RKIND), pointer:: &
        config_forcing_ramp, &
        daysSinceStartOfSim

    integer, pointer :: &
         nCellsSolve

    integer :: &
         iCell

    call MPAS_pool_get_config(block % configs, "config_time_varying_atmospheric_forcing_ramp", config_forcing_ramp)

    call MPAS_pool_get_subpool(block % structs, "mesh", mesh)
    call MPAS_pool_get_subpool(block % structs, "forcing", forcingPool)
    call MPAS_pool_get_subpool(block % structs, "timeVaryingForcing", timeVaryingForcingPool)
    call MPAS_pool_get_subpool(block % structs, "diagnostics", diagnosticsPool)

    call MPAS_pool_get_dimension(mesh, "nCellsSolve", nCellsSolve)

    call MPAS_pool_get_array(diagnosticsPool, "daysSinceStartOfSim", daysSinceStartOfSim)
    call MPAS_pool_get_array(timeVaryingForcingPool, "windSpeedU", windSpeedU)
    call MPAS_pool_get_array(timeVaryingForcingPool, "windSpeedV", windSpeedV)
    call MPAS_pool_get_array(timeVaryingForcingPool, "windSpeedMagnitude", windSpeedMagnitude)
    call MPAS_pool_get_array(forcingPool, "windStressZonal", windStressZonal)
    call MPAS_pool_get_array(forcingPool, "windStressMeridional", windStressMeridional)

    call MPAS_pool_get_array(timeVaryingForcingPool, "atmosPressure", atmosPressure)
    call MPAS_pool_get_array(forcingPool, "atmosphericPressure", atmosphericPressure)

    rhoAir = 1.225_RKIND

    ramp = tanh((2.0_RKIND*daysSinceStartOfSim)/config_forcing_ramp)

    do iCell = 1, nCellsSolve

       windSpeedMagnitude(iCell) = sqrt(windSpeedU(iCell)**2 + windSpeedV(iCell)**2)

       windStressCoefficient = (0.75_RKIND + 0.067_RKIND * windSpeedMagnitude(iCell)) * 0.001_RKIND * rhoAir  ! Garratt 1977

       windStressZonal(iCell) = ramp * windSpeedU(iCell) * windSpeedMagnitude(iCell) * windStressCoefficient
       windStressMeridional(iCell) = ramp * windSpeedV(iCell) * windSpeedMagnitude(iCell) * windStressCoefficient

    enddo

    atmosphericPressure = atmosPressure

  end subroutine post_atmospheric_forcing

!-----------------------------------------------------------------------
! coupler fluxes initialization
!-----------------------------------------------------------------------
!
! Not needed now, but may be for cumlative forcing i.e. river flux
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_reset_coupler_fluxes
!
!> \brief
!> \author Steven Brus, LANL
!> \date
!> \details
!>
!
!-----------------------------------------------------------------------

!  subroutine ocn_reset_coupler_fluxes(domain)
!
!    type(domain_type) :: domain
!
!    logical, pointer :: &
!         config_use_column_package
!
!    call MPAS_pool_get_config(domain % blocklist % configs, "config_use_column_package", config_use_column_package)
!
!    if (config_use_column_package) then
!
!       call reset_atmospheric_coupler_fluxes(domain)
!
!    endif
!
!  end subroutine ocn_reset_coupler_fluxes

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  reset_atmospheric_coupler_fluxes
!
!> \brief
!> \author Steven Brus, LANL
!> \date
!> \details
!>
!
!-----------------------------------------------------------------------


!  subroutine reset_atmospheric_coupler_fluxes(domain)
!
!    type(domain_type) :: domain
!
!    type(block_type), pointer :: block
!
!    type(MPAS_pool_type), pointer :: &
!         atmosFluxes, &
!         shortwave, &
!         velocitySolver, &
!         forcingPool, &
!         biogeochemistry
!
!    real(kind=RKIND), dimension(:), pointer :: &
!         sensibleHeatFlux, &
!         latentHeatFlux, &
!         absorbedShortwaveFlux, &
!         longwaveUp, &
!         evaporativeWaterFlux, &
!         airStressCellU, &
!         airStressCellV, &
!         atmosReferenceSpeed10m, &
!         atmosReferenceTemperature2m, &
!         atmosReferenceHumidity2m
!
!    real(kind=RKIND), dimension(:,:), pointer :: &
!         atmosBioFluxes, &
!         atmosBlackCarbonFlux, &
!         atmosDustFlux
!
!    logical, pointer :: &
!         config_use_column_biogeochemistry
!
!    block => domain % blocklist
!    do while (associated(block))
!
!       ! physics
!       call MPAS_pool_get_subpool(block % structs, "atmos_fluxes", atmosFluxes)
!       call MPAS_pool_get_subpool(block % structs, "shortwave", shortwave)
!       call MPAS_pool_get_subpool(block % structs, "velocity_solver", velocitySolver)
!       call MPAS_pool_get_subpool(block % structs, "atmos_coupling", forcingPool)
!
!       call MPAS_pool_get_array(atmosFluxes, "sensibleHeatFlux", sensibleHeatFlux)
!       call MPAS_pool_get_array(atmosFluxes, "latentHeatFlux", latentHeatFlux)
!       call MPAS_pool_get_array(atmosFluxes, "longwaveUp", longwaveUp)
!       call MPAS_pool_get_array(atmosFluxes, "evaporativeWaterFlux", evaporativeWaterFlux)
!
!       call MPAS_pool_get_array(velocitySolver, "airStressCellU", airStressCellU)
!       call MPAS_pool_get_array(velocitySolver, "airStressCellV", airStressCellV)
!
!       call MPAS_pool_get_array(shortwave, "absorbedShortwaveFlux", absorbedShortwaveFlux)
!
!       call MPAS_pool_get_array(forcingPool, 'atmosReferenceSpeed10m', atmosReferenceSpeed10m)
!       call MPAS_pool_get_array(forcingPool, 'atmosReferenceTemperature2m', atmosReferenceTemperature2m)
!       call MPAS_pool_get_array(forcingPool, 'atmosReferenceHumidity2m', atmosReferenceHumidity2m)
!
!       sensibleHeatFlux            = 0.0_RKIND
!       latentHeatFlux              = 0.0_RKIND
!       longwaveUp                  = 0.0_RKIND
!       evaporativeWaterFlux        = 0.0_RKIND
!
!       airStressCellU              = 0.0_RKIND
!       airStressCellV              = 0.0_RKIND
!
!       absorbedShortwaveFlux       = 0.0_RKIND
!
!       atmosReferenceSpeed10m      = 0.0_RKIND
!       atmosReferenceTemperature2m = 0.0_RKIND
!       atmosReferenceHumidity2m    = 0.0_RKIND
!
!
!       block => block % next
!    end do
!
!  end subroutine reset_atmospheric_coupler_fluxes

!-----------------------------------------------------------------------
! restart
!-----------------------------------------------------------------------
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_time_varying_forcing_write_restart_times
!
!> \brief
!> \author Steven Brus, LANL
!> \date
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine ocn_time_varying_forcing_write_restart_times(domain)

    type(domain_type) :: domain

    logical, pointer :: &
         config_use_atmospheric_forcing

    call MPAS_pool_get_config(domain % configs, "config_use_time_varying_atmospheric_forcing", config_use_atmospheric_forcing)

    if (config_use_atmospheric_forcing) then

       call MPAS_forcing_write_restart_times(forcingGroupHead)

    endif

  end subroutine ocn_time_varying_forcing_write_restart_times

!-----------------------------------------------------------------------

end module ocn_time_varying_forcing
