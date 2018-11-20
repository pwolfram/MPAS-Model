!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_forcing
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

module ocn_forcing

  use mpas_derived_types
  use mpas_pool_routines
  use mpas_timekeeping
  use mpas_forcing
  use mpas_stream_manager
  use mpas_log, only: mpas_log_write

  implicit none

  private
  public :: &
       ocn_forcing_init, &
       ocn_forcing_get, &
       ocn_forcing_write_restart_times, &
       ocn_reset_coupler_fluxes,  &
       post_atmospheric_coupling,  &
       post_atmospheric_forcing

  type (MPAS_forcing_group_type), pointer :: ocnForcingGroups

contains

!-----------------------------------------------------------------------
! initialization
!-----------------------------------------------------------------------

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_forcing_init
!
!> \brief Driver for ocean core forcing initialization
!> \author Steven Brus, LANL
!> \date
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine ocn_forcing_init(domain)

    type (domain_type) :: domain

    logical, pointer :: &
         config_use_forcing

    call MPAS_pool_get_config(domain % configs, "config_use_forcing", config_use_forcing)

    if (config_use_forcing) then

       ! init the atmospheric forcing
       call init_atmospheric_forcing(domain)

       ! init other forcings, to be added later as needed
       !call init_seaice_forcing(domain)
       !  .
       !  .
       !  .
      

    endif

  end subroutine ocn_forcing_init

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

    call MPAS_pool_get_config(domain % configs, "config_atmospheric_forcing_type", config_atmospheric_forcing_type)

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

    real(kind=RKIND), pointer :: &
         config_dt

    character(len=strKIND), pointer :: &
         config_forcing_start_time, &
         config_forcing_cycle_start, &
         config_forcing_cycle_duration

    logical, pointer :: &
         config_do_restart

    character(len=strKIND) :: &
         forcingIntervalSixHourly, &
         forcingReferenceTimeSixHourly, &
         forcingIntervalMonthly, &
         forcingReferenceTimeMonthly

    ! get atmospheric forcing configuration options
    call MPAS_pool_get_config(domain % configs, "config_forcing_start_time", config_forcing_start_time)
    call MPAS_pool_get_config(domain % configs, "config_dt", config_dt)
    call MPAS_pool_get_config(domain % configs, "config_forcing_cycle_start", config_forcing_cycle_start)
    call MPAS_pool_get_config(domain % configs, "config_forcing_cycle_duration", config_forcing_cycle_duration)
    call MPAS_pool_get_config(domain % configs, "config_do_restart", config_do_restart)

    ! create the six hourly forcing group
    call MPAS_forcing_init_group(&                
         ocnForcingGroups, &                     ! forcingGroupHead
         "ocn_atmospheric_forcing", &            ! forcingGroupName
         domain, &                               ! domain
         config_forcing_start_time, &            ! startTimeStr
         config_forcing_cycle_start, &           ! forcingCycleStart
         config_forcing_cycle_duration, &        ! forcingCycleDuration
         config_do_restart, &                    ! restart
         .false.)                                ! forcingCycleStartInclusive

    forcingIntervalSixHourly = "06:00:00"
    forcingReferenceTimeSixHourly = "2000-01-01_00:00:00"


    call MPAS_forcing_init_field(&               
         domain % streamManager, &               ! streamManager
         ocnForcingGroups, &                     ! forcingGroupHead
         "ocn_atmospheric_forcing", &            ! forcingGroupName
         "uWind", &                              ! forcingName
         "atmospheric_forcing", &                ! forcingStreamID
         "atmos_coupling", &                     ! poolname
         "uWind", &                              ! fieldname
         "linear", &                             ! interpolationType
         forcingReferenceTimeSixHourly, &        ! forcingReferenceTimeStr
         forcingIntervalSixHourly, &             ! forcingIntervalStr
         "next")                                 ! forcingInitializationType

    call MPAS_forcing_init_field(&
         domain % streamManager, &
         ocnForcingGroups, &
         "ocn_atmospheric_forcing", &
         "vWind", &
         "atmospheric_forcing", &
         "atmos_coupling", &
         "vWind", &
         "linear", &
         forcingReferenceTimeSixHourly, &
         forcingIntervalSixHourly, &
         "next")

    call MPAS_forcing_init_field(&
         domain % streamManager, &
         ocnForcingGroups, &
         "ocn_atmospheric_forcing", &
         "atmosphericPressure", &
         "atmospheric_forcing", &
         "atmos_coupling", &
         "atmospericPressure", &
         "linear", &
         forcingReferenceTimeSixHourly, &
         forcingIntervalSixHourly, &
         "next")

    call MPAS_forcing_init_field_data(&
         ocnForcingGroups, &
         "ocn_atmospheric_forcing", &
         domain % streamManager, &
         config_do_restart, &
         .false.)

  end subroutine init_atmospheric_forcing_WINDPRES

!-----------------------------------------------------------------------
! runtime
!-----------------------------------------------------------------------

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_forcing
!
!> \brief Retrieve forcing data during time stepping
!> \author Steven Brus, LANL
!> \date 2013-2014
!> \details
!>  This routine calls the MPAS_forcing routine that will perform the
!>  forcing data aquisition and interpolation during timestepping
!
!-----------------------------------------------------------------------

  subroutine ocn_forcing_get(&
       streamManager, &
       domain, &
       simulationClock, &
       firstTimeStep)

    type (MPAS_streamManager_type), intent(inout) :: streamManager

    type (domain_type) :: domain

    type (MPAS_clock_type) :: simulationClock

    logical, intent(in) :: &
         firstTimeStep

    logical, pointer :: &
         config_use_forcing

    call MPAS_pool_get_config(domain % configs, "config_use_forcing", config_use_forcing)

    if (config_use_forcing) then

       call atmospheric_forcing(&
            streamManager, &
            domain, &
            simulationClock)

       ! call other forcings, to be added if needed
       ! .
       ! .
       ! .

    endif

  end subroutine ocn_forcing_get

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

    type (block_type), pointer :: block

    real(kind=RKIND), pointer :: &
         config_dt

    character(len=strKIND), pointer :: &
         config_atmospheric_forcing_type

    ! configurations
    call mpas_pool_get_config(domain % configs, 'config_dt', config_dt)
    call mpas_pool_get_config(domain % configs, 'config_atmospheric_forcing_type', config_atmospheric_forcing_type)

    ! use the forcing layer to get data
    if (trim(config_atmospheric_forcing_type) == "WINDPRES") then

       call MPAS_forcing_get_forcing(&
            ocnForcingGroups, &
            "ocn_atmospheric_forcing", &
            streamManager, &
            config_dt)

       call MPAS_forcing_get_forcing_time(&
            ocnForcingGroups, &
            "ocn_atmospheric_forcing", &
            currentForcingTime)

    endif

    block => domain % blocklist
    do while (associated(block))

       ! perform post coupling operations
       call post_atmospheric_coupling(block)

       ! perform post forcing
       call post_atmospheric_forcing(block)

       block => block % next
    end do

  end subroutine atmospheric_forcing

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  post_atmospheric_coupling
!
!> \brief
!> \author Steven Brus, LANL
!> \date
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine post_atmospheric_coupling(block)

    use ocn_mesh, only: &
         ocn_latlon_vector_rotation_forward

    type (block_type), pointer :: block

    type (mpas_pool_type), pointer :: &
         mesh, &
         atmosCoupling, &
         atmosForcing

    real(kind=RKIND), dimension(:), pointer :: &
         uWind, &
         vWind, &
         windSpeed, &
         latCell, &
         lonCell, &
         xCell, &
         yCell, &
         zCell

    real(kind=RKIND), pointer :: &
         sphere_radius

    logical, pointer :: &
         config_rotate_cartesian_grid

    integer, pointer :: &
         nCellsSolve

    integer :: &
         iCell

    call MPAS_pool_get_subpool(block % structs, "mesh", mesh)
    call MPAS_pool_get_subpool(block % structs, "atmos_coupling", atmosCoupling)
    call MPAS_pool_get_subpool(block % structs, "atmos_forcing", atmosForcing)

    call MPAS_pool_get_config(block % configs, "config_rotate_cartesian_grid", config_rotate_cartesian_grid)

    call MPAS_pool_get_config(mesh, "sphere_radius", sphere_radius)
    call MPAS_pool_get_dimension(mesh, "nCellsSolve", nCellsSolve)

    call MPAS_pool_get_array(mesh, "latCell", latCell)
    call MPAS_pool_get_array(mesh, "lonCell", lonCell)
    call MPAS_pool_get_array(mesh, "xCell", xCell)
    call MPAS_pool_get_array(mesh, "yCell", yCell)
    call MPAS_pool_get_array(mesh, "zCell", zCell)

    call MPAS_pool_get_array(atmosCoupling, "uWind", uWind)
    call MPAS_pool_get_array(atmosCoupling, "vWind", vWind)

    call MPAS_pool_get_array(atmosForcing, "windSpeed", windSpeed)

    do iCell = 1, nCellsSolve

       ! wind speed
       windSpeed(iCell) = sqrt(uWind(iCell)**2 + vWind(iCell)**2)

       ! rotate velocities from geographical to local
       call ocn_latlon_vector_rotation_forward(&
            uAirVelocity(iCell), &
            vAirVelocity(iCell), &
            uAirVelocity(iCell), &
            vAirVelocity(iCell), &
            latCell(iCell), &
            lonCell(iCell), &
            xCell(iCell), &
            yCell(iCell), &
            zCell(iCell), &
            sphere_radius, &
            config_rotate_cartesian_grid)

    enddo ! iCell

  end subroutine post_atmospheric_coupling

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

    type (mpas_pool_type), pointer :: &
         mesh, &
         atmosCoupling, &
         atmosForcing

    real(kind=RKIND), dimension(:), pointer :: &
         uAirVelocity, &
         vAirVelocity, &
         windSpeed, &
         uAirStress, &
         vAirStress

    real(kind=RKIND) :: &
         airStressCoefficient

    integer, pointer :: &
         nCellsSolve

    integer :: &
         iCell

    call MPAS_pool_get_subpool(block % structs, "mesh", mesh)
    call MPAS_pool_get_subpool(block % structs, "atmos_coupling", atmosCoupling)
    call MPAS_pool_get_subpool(block % structs, "atmos_forcing", atmosForcing)

    call MPAS_pool_get_dimension(mesh, "nCellsSolve", nCellsSolve)

    call MPAS_pool_get_array(atmosCoupling, "airDensity", airDensity)
    call MPAS_pool_get_array(atmosCoupling, "uAirVelocity", uAirVelocity)
    call MPAS_pool_get_array(atmosCoupling, "vAirVelocity", vAirVelocity)

    call MPAS_pool_get_array(atmosForcing, "windSpeed", windSpeed)
    call MPAS_pool_get_array(atmosForcing, "sensibleTransferCoefficient", sensibleTransferCoefficient)
    call MPAS_pool_get_array(atmosForcing, "latentTransferCoefficient", latentTransferCoefficient)
    call MPAS_pool_get_array(atmosForcing, "uAirStress", uAirStress)
    call MPAS_pool_get_array(atmosForcing, "vAirStress", vAirStress)

    do iCell = 1, nCellsSolve

       ! air stresses
       airStressCoefficient = 0.0012_RKIND * airDensity(iCell) * windSpeed(iCell)

       uAirStress(iCell) = uAirVelocity(iCell) * airStressCoefficient
       vAirStress(iCell) = vAirVelocity(iCell) * airStressCoefficient

    enddo ! iCell

  end subroutine post_atmospheric_forcing

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  get_seconds_today
!
!> \brief
!> \author Steven Brus, LANL
!> \date
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine get_seconds_today(&
       currentTime, &
       secondsToday, &
       dayOfYear)

    type(MPAS_time_type), intent(in) :: &
         currentTime

    real(kind=RKIND), intent(out) :: &
         secondsToday

    integer, intent(out) :: &
         dayOfYear

    integer :: &
         H, M, S, S_d, S_n

    call mpas_get_time(currentTime, DoY=dayOfYear, H=H, M=M, S=S, S_n=S_n, S_d=S_d)

    secondsToday = real(H,RKIND)*3600.0_RKIND + &
                   real(M,RKIND)*60.0_RKIND + &
                   real(S,RKIND) + &
                   real(S_n,RKIND)/real(S_d,RKIND)

  end subroutine get_seconds_today

!-----------------------------------------------------------------------
! coupler fluxes initialization
!-----------------------------------------------------------------------

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

  subroutine ocn_reset_coupler_fluxes(domain)

    type(domain_type) :: domain

    logical, pointer :: &
         config_use_column_package

    call MPAS_pool_get_config(domain % blocklist % configs, "config_use_column_package", config_use_column_package)

    if (config_use_column_package) then

       call reset_atmospheric_coupler_fluxes(domain)
       call reset_ocean_coupler_fluxes(domain)

    endif

  end subroutine ocn_reset_coupler_fluxes

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

  subroutine reset_atmospheric_coupler_fluxes(domain)

    type(domain_type) :: domain

    type(block_type), pointer :: block

    type(MPAS_pool_type), pointer :: &
         atmosFluxes, &
         shortwave, &
         velocitySolver, &
         atmosCoupling, &
         biogeochemistry

    real(kind=RKIND), dimension(:), pointer :: &
         sensibleHeatFlux, &
         latentHeatFlux, &
         absorbedShortwaveFlux, &
         longwaveUp, &
         evaporativeWaterFlux, &
         airStressCellU, &
         airStressCellV, &
         atmosReferenceSpeed10m, &
         atmosReferenceTemperature2m, &
         atmosReferenceHumidity2m

    real(kind=RKIND), dimension(:,:), pointer :: &
         atmosBioFluxes, &
         atmosBlackCarbonFlux, &
         atmosDustFlux

    logical, pointer :: &
         config_use_column_biogeochemistry

    block => domain % blocklist
    do while (associated(block))

       ! physics
       call MPAS_pool_get_subpool(block % structs, "atmos_fluxes", atmosFluxes)
       call MPAS_pool_get_subpool(block % structs, "shortwave", shortwave)
       call MPAS_pool_get_subpool(block % structs, "velocity_solver", velocitySolver)
       call MPAS_pool_get_subpool(block % structs, "atmos_coupling", atmosCoupling)

       call MPAS_pool_get_array(atmosFluxes, "sensibleHeatFlux", sensibleHeatFlux)
       call MPAS_pool_get_array(atmosFluxes, "latentHeatFlux", latentHeatFlux)
       call MPAS_pool_get_array(atmosFluxes, "longwaveUp", longwaveUp)
       call MPAS_pool_get_array(atmosFluxes, "evaporativeWaterFlux", evaporativeWaterFlux)

       call MPAS_pool_get_array(velocitySolver, "airStressCellU", airStressCellU)
       call MPAS_pool_get_array(velocitySolver, "airStressCellV", airStressCellV)

       call MPAS_pool_get_array(shortwave, "absorbedShortwaveFlux", absorbedShortwaveFlux)

       call MPAS_pool_get_array(atmosCoupling, 'atmosReferenceSpeed10m', atmosReferenceSpeed10m)
       call MPAS_pool_get_array(atmosCoupling, 'atmosReferenceTemperature2m', atmosReferenceTemperature2m)
       call MPAS_pool_get_array(atmosCoupling, 'atmosReferenceHumidity2m', atmosReferenceHumidity2m)

       sensibleHeatFlux            = 0.0_RKIND
       latentHeatFlux              = 0.0_RKIND
       longwaveUp                  = 0.0_RKIND
       evaporativeWaterFlux        = 0.0_RKIND

       airStressCellU              = 0.0_RKIND
       airStressCellV              = 0.0_RKIND

       absorbedShortwaveFlux       = 0.0_RKIND

       atmosReferenceSpeed10m      = 0.0_RKIND
       atmosReferenceTemperature2m = 0.0_RKIND
       atmosReferenceHumidity2m    = 0.0_RKIND

       ! biogeochemistry
       call MPAS_pool_get_config(block % configs, "config_use_column_biogeochemistry", config_use_column_biogeochemistry)

       if (config_use_column_biogeochemistry) then

          call MPAS_pool_get_subpool(block % structs, "biogeochemistry", biogeochemistry)

          call MPAS_pool_get_array(biogeochemistry, "atmosBioFluxes", atmosBioFluxes)
          call MPAS_pool_get_array(biogeochemistry, "atmosBlackCarbonFlux", atmosBlackCarbonFlux)
          call MPAS_pool_get_array(biogeochemistry, "atmosDustFlux", atmosDustFlux)

          atmosBioFluxes              = 0.0_RKIND
          atmosBlackCarbonFlux        = 0.0_RKIND
          atmosDustFlux               = 0.0_RKIND

       endif

       block => block % next
    end do

  end subroutine reset_atmospheric_coupler_fluxes

!-----------------------------------------------------------------------
! restart
!-----------------------------------------------------------------------
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_forcing_write_restart_times
!
!> \brief
!> \author Steven Brus, LANL
!> \date
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine ocn_forcing_write_restart_times(domain)

    type(domain_type) :: domain

    logical, pointer :: &
         config_use_forcing

    call MPAS_pool_get_config(domain % configs, "config_use_forcing", config_use_forcing)

    if (config_use_forcing) then

       call MPAS_forcing_write_restart_times(ocnForcingGroups)

    endif

  end subroutine ocn_forcing_write_restart_times

!-----------------------------------------------------------------------

end module ocn_forcing
