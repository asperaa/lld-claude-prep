"""
Parking Lot — Low-Level Design
================================
Full implementation consolidating all code from the study guide.

Sections in this file:
  1. Enums            — VehicleType, SpotSize, SpotType, SpotStatus
  2. Exceptions       — domain-specific errors
  3. Vehicle          — base class + concrete subclasses + VehicleFactory
  4. ParkingSpot      — spot state machine, assignment, release
  5. ParkingFloor     — groups spots, exposes availability
  6. Ticket           — transaction record, duration helper
  7. FeeCalculator    — abstract base + HourlyFeeCalculator (Template Method)
  8. ParkingStrategy  — abstract base + NearestSpotStrategy
                        + HandicappedFirstStrategy
  9. Observer         — SpotStatusObserver + DisplayBoardObserver
                        + AvailabilityCounterObserver
 10. ParkingLot       — orchestrator / Singleton / Facade
 11. Demo             — end-to-end smoke test (run this file directly)
"""

from __future__ import annotations

import math
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional


# ─────────────────────────────────────────────
# 1. ENUMS
# ─────────────────────────────────────────────

class VehicleType(Enum):
    MOTORCYCLE = auto()
    CAR        = auto()
    TRUCK      = auto()


class SpotSize(Enum):
    SMALL  = auto()   # motorcycles
    MEDIUM = auto()   # cars
    LARGE  = auto()   # trucks


class SpotType(Enum):
    STANDARD    = auto()
    HANDICAPPED = auto()
    EV_CHARGING = auto()


class SpotStatus(Enum):
    FREE     = auto()
    OCCUPIED = auto()
    RESERVED = auto()
    BLOCKED  = auto()   # maintenance


# Maps vehicle type → the spot size it requires by default.
# Lives here as shared data used by both ParkingSpot.can_fit()
# and strategies — single source of truth for the vehicle↔size policy.
DEFAULT_SIZE_MAP: dict[VehicleType, SpotSize] = {
    VehicleType.MOTORCYCLE: SpotSize.SMALL,
    VehicleType.CAR:        SpotSize.MEDIUM,
    VehicleType.TRUCK:      SpotSize.LARGE,
}

# Valid spot-status transitions — State Machine as data, not code.
# A transition not listed here is forbidden; no special-case logic needed.
VALID_SPOT_TRANSITIONS: dict[SpotStatus, list[SpotStatus]] = {
    SpotStatus.FREE:     [SpotStatus.OCCUPIED, SpotStatus.RESERVED, SpotStatus.BLOCKED],
    SpotStatus.OCCUPIED: [SpotStatus.FREE],
    SpotStatus.RESERVED: [SpotStatus.OCCUPIED, SpotStatus.FREE],
    SpotStatus.BLOCKED:  [SpotStatus.FREE],   # only ops can unblock
}


# ─────────────────────────────────────────────
# 2. EXCEPTIONS
# ─────────────────────────────────────────────

class ParkingLotError(Exception):
    """Base for all domain errors."""


class SpotUnavailableError(ParkingLotError):
    def __init__(self, spot_id: str):
        super().__init__(f"Spot {spot_id!r} is not available for assignment.")


class InvalidSpotTransitionError(ParkingLotError):
    def __init__(self, from_status: SpotStatus, to_status: SpotStatus):
        super().__init__(
            f"Invalid spot transition: {from_status.name} → {to_status.name}"
        )


class LotFullError(ParkingLotError):
    def __init__(self):
        super().__init__("No available spot found for this vehicle.")


class AlreadyParkedError(ParkingLotError):
    def __init__(self, plate: str):
        super().__init__(f"Vehicle {plate!r} is already parked.")


class TicketNotFoundError(ParkingLotError):
    def __init__(self, plate: str):
        super().__init__(f"No active ticket found for plate {plate!r}.")


# ─────────────────────────────────────────────
# 3. VEHICLE
# ─────────────────────────────────────────────

class Vehicle:
    """
    Value object: identity comes from license_plate.
    Deliberately minimal — no owner, no registration for LLD scope.
    """

    def __init__(self, license_plate: str, vehicle_type: VehicleType):
        self.license_plate = license_plate
        self.vehicle_type  = vehicle_type

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}"
                f"({self.license_plate!r})")


class Motorcycle(Vehicle):
    def __init__(self, license_plate: str):
        super().__init__(license_plate, VehicleType.MOTORCYCLE)


class Car(Vehicle):
    def __init__(self, license_plate: str):
        super().__init__(license_plate, VehicleType.CAR)


class Truck(Vehicle):
    def __init__(self, license_plate: str):
        super().__init__(license_plate, VehicleType.TRUCK)


class VehicleFactory:
    """
    Simple Factory — centralises the string→class mapping so no
    caller ever writes their own if/else to create a Vehicle.
    Dict-based: adding a new type is one line, zero edits to callers.
    """

    _REGISTRY: dict[str, type[Vehicle]] = {
        "MOTORCYCLE": Motorcycle,
        "CAR":        Car,
        "TRUCK":      Truck,
    }

    @staticmethod
    def create(license_plate: str, type_str: str) -> Vehicle:
        cls = VehicleFactory._REGISTRY.get(type_str.upper())
        if cls is None:
            known = list(VehicleFactory._REGISTRY)
            raise ValueError(
                f"Unknown vehicle type {type_str!r}. Known: {known}"
            )
        return cls(license_plate)


# ─────────────────────────────────────────────
# 4. PARKING SPOT
# ─────────────────────────────────────────────

class ParkingSpot:
    """
    Core domain object. Owns its own state transitions (Tell Don't Ask).
    Notifies registered observers whenever status changes.
    """

    def __init__(
        self,
        spot_id:      str,
        size:         SpotSize,
        spot_type:    SpotType,
        floor_number: int,
        status:       SpotStatus = SpotStatus.FREE,
    ):
        self.spot_id       = spot_id
        self.size          = size
        self.spot_type     = spot_type
        self.floor_number  = floor_number
        self._status       = status
        self._vehicle:     Optional[Vehicle] = None
        self._observers:   list[SpotStatusObserver] = []

    # ── status property — all writes go through _transition() ──

    @property
    def status(self) -> SpotStatus:
        return self._status

    def _transition(self, new_status: SpotStatus) -> None:
        allowed = VALID_SPOT_TRANSITIONS.get(self._status, [])
        if new_status not in allowed:
            raise InvalidSpotTransitionError(self._status, new_status)
        self._status = new_status
        for obs in self._observers:
            obs.on_spot_changed(self, new_status)

    # ── public API ──

    def is_available(self) -> bool:
        """Single method encapsulates the definition of 'available'.
        If the rule changes (e.g. also check maintenance schedule),
        only this method changes — all callers get the update for free."""
        return self._status == SpotStatus.FREE

    def can_fit(self, vehicle: Vehicle) -> bool:
        """Physical fit check. Rule lives on the spot because the spot's
        size is its own attribute. Strategies call this — they never
        re-implement the size logic themselves."""
        return self.size == DEFAULT_SIZE_MAP[vehicle.vehicle_type]

    def assign_vehicle(self, vehicle: Vehicle) -> None:
        """Atomically checks availability, records the vehicle, and
        transitions status. All in one call — no two-step race window."""
        if not self.is_available():
            raise SpotUnavailableError(self.spot_id)
        self._vehicle = vehicle
        self._transition(SpotStatus.OCCUPIED)

    def release_vehicle(self) -> Vehicle:
        """Returns the vehicle so the caller can close the ticket
        without needing a separate read before calling release."""
        vehicle = self._vehicle
        self._vehicle = None
        self._transition(SpotStatus.FREE)
        return vehicle  # type: ignore[return-value]

    def add_observer(self, obs: SpotStatusObserver) -> None:
        self._observers.append(obs)

    def __repr__(self) -> str:
        return (f"ParkingSpot({self.spot_id!r}, "
                f"{self.size.name}, {self.spot_type.name}, "
                f"floor={self.floor_number}, {self._status.name})")


# ─────────────────────────────────────────────
# 5. PARKING FLOOR
# ─────────────────────────────────────────────

class ParkingFloor:
    """
    Groups spots. Exists so you can query floor-level availability
    without knowing every spot in the building.
    Without ParkingFloor, ParkingLot holds a flat list and
    floor-level features (display boards, nearest-floor search) are lost.
    """

    def __init__(self, floor_number: int, spots: list[ParkingSpot]):
        self.floor_number = floor_number
        self.spots        = spots

    def available_spots(
        self,
        size:       Optional[SpotSize] = None,
        spot_type:  Optional[SpotType] = None,
    ) -> list[ParkingSpot]:
        return [
            s for s in self.spots
            if s.is_available()
            and (size      is None or s.size      == size)
            and (spot_type is None or s.spot_type == spot_type)
        ]

    def free_count(self) -> int:
        return sum(1 for s in self.spots if s.is_available())

    def __repr__(self) -> str:
        return (f"ParkingFloor({self.floor_number}, "
                f"free={self.free_count()}/{len(self.spots)})")


# ─────────────────────────────────────────────
# 6. TICKET
# ─────────────────────────────────────────────

@dataclass
class Ticket:
    """
    Immutable transaction record.  entry_time is set at creation and
    never mutated — it is a fact about when the transaction began.
    exit_time and fee start as None (not yet determined), which is
    semantically distinct from 0 or a placeholder.
    """
    vehicle:    Vehicle
    spot:       ParkingSpot
    entry_time: datetime          = field(default_factory=datetime.now)
    ticket_id:  str               = field(default_factory=lambda: str(uuid.uuid4()))
    exit_time:  Optional[datetime] = field(default=None)
    fee:        Optional[float]    = field(default=None)

    def duration_hours(self) -> float:
        """Uses exit_time if set; falls back to now() for in-progress tickets.
        Lives on Ticket because duration is derived from Ticket's own data —
        no external class needs to know how to compute it."""
        end   = self.exit_time or datetime.now()
        delta = end - self.entry_time
        return delta.total_seconds() / 3600

    def is_closed(self) -> bool:
        return self.exit_time is not None

    def __repr__(self) -> str:
        status = "closed" if self.is_closed() else "active"
        return (f"Ticket({self.ticket_id[:8]}… "
                f"{self.vehicle.license_plate!r} "
                f"spot={self.spot.spot_id!r} [{status}])")


# ─────────────────────────────────────────────
# 7. FEE CALCULATOR  (Template Method pattern)
# ─────────────────────────────────────────────

class FeeCalculator(ABC):
    """
    Template Method: calculate() is the fixed skeleton (get duration,
    round up, multiply by rate). get_rate() is the variable step —
    subclasses override only the part that changes.
    New pricing model = new subclass, zero changes here.
    """

    def calculate(self, ticket: Ticket) -> float:
        hours = math.ceil(ticket.duration_hours())
        # Minimum 1 hour charge even if parked for 5 minutes —
        # ceil(0.08) = 1, which is correct for most real parking lots.
        hours = max(1, hours)
        rate  = self.get_rate(ticket.vehicle.vehicle_type)
        return float(hours * rate)

    @abstractmethod
    def get_rate(self, vehicle_type: VehicleType) -> float:
        """Return the per-hour rate (in ₹) for this vehicle type."""
        ...


class HourlyFeeCalculator(FeeCalculator):
    """Weekday standard rates."""

    RATES: dict[VehicleType, float] = {
        VehicleType.MOTORCYCLE: 20.0,
        VehicleType.CAR:        50.0,
        VehicleType.TRUCK:     100.0,
    }

    def get_rate(self, vehicle_type: VehicleType) -> float:
        return self.RATES[vehicle_type]


class WeekendFeeCalculator(FeeCalculator):
    """Weekend premium rates — only get_rate() differs."""

    RATES: dict[VehicleType, float] = {
        VehicleType.MOTORCYCLE: 30.0,
        VehicleType.CAR:        80.0,
        VehicleType.TRUCK:     150.0,
    }

    def get_rate(self, vehicle_type: VehicleType) -> float:
        return self.RATES[vehicle_type]


# ─────────────────────────────────────────────
# 8. PARKING STRATEGY  (Strategy pattern)
# ─────────────────────────────────────────────

class ParkingStrategy(ABC):
    """
    Abstract base for all spot-selection algorithms.
    Returns None (not raises) when no spot is found — "no spot" is
    a normal business situation, not an error. The orchestrator
    (ParkingLot) decides what to do with None.
    """

    @abstractmethod
    def find_spot(
        self,
        floors:  list[ParkingFloor],
        vehicle: Vehicle,
    ) -> Optional[ParkingSpot]:
        ...


class NearestSpotStrategy(ParkingStrategy):
    """
    Scan floors in order, return the first available spot that fits.
    'Nearest' here means nearest floor/index — no GPS coordinates needed
    for LLD scope.
    """

    def find_spot(
        self,
        floors:  list[ParkingFloor],
        vehicle: Vehicle,
    ) -> Optional[ParkingSpot]:
        for floor in floors:
            for spot in floor.spots:
                if spot.is_available() and spot.can_fit(vehicle):
                    return spot
        return None


class HandicappedFirstStrategy(ParkingStrategy):
    """
    Prioritise HANDICAPPED spots, then fall back to nearest regular spot.
    Composes with NearestSpotStrategy instead of duplicating its logic —
    adding a new type of priority is additive, never duplicative.
    """

    def __init__(self):
        self._fallback = NearestSpotStrategy()

    def find_spot(
        self,
        floors:  list[ParkingFloor],
        vehicle: Vehicle,
    ) -> Optional[ParkingSpot]:
        # Pass 1: handicapped spots only
        for floor in floors:
            for spot in floor.spots:
                if (spot.is_available()
                        and spot.can_fit(vehicle)
                        and spot.spot_type == SpotType.HANDICAPPED):
                    return spot
        # Pass 2: any regular spot
        return self._fallback.find_spot(floors, vehicle)


class EVPriorityStrategy(ParkingStrategy):
    """
    Prioritise EV_CHARGING spots for all vehicles, then fall back.
    Demonstrates that extending with a new strategy requires zero
    changes to ParkingLot or any existing strategy.
    """

    def __init__(self):
        self._fallback = NearestSpotStrategy()

    def find_spot(
        self,
        floors:  list[ParkingFloor],
        vehicle: Vehicle,
    ) -> Optional[ParkingSpot]:
        for floor in floors:
            for spot in floor.spots:
                if (spot.is_available()
                        and spot.can_fit(vehicle)
                        and spot.spot_type == SpotType.EV_CHARGING):
                    return spot
        return self._fallback.find_spot(floors, vehicle)


# ─────────────────────────────────────────────
# 9. OBSERVER  (Observer pattern)
# ─────────────────────────────────────────────

class SpotStatusObserver(ABC):
    """
    Any component that wants to react to a spot changing status
    implements this interface and registers itself on the spot.
    ParkingSpot never imports DisplayBoard or any concrete observer —
    coupling flows only downward (spot → observer interface).
    """

    @abstractmethod
    def on_spot_changed(
        self,
        spot:       ParkingSpot,
        new_status: SpotStatus,
    ) -> None:
        ...


class DisplayBoardObserver(SpotStatusObserver):
    """
    Tracks free-spot counts per floor.
    In production this would push updates to an LED display API.
    Here it just maintains an in-memory count readable via get_count().
    """

    def __init__(self, floors: list[ParkingFloor]):
        # Initialise from current floor state
        self._counts: dict[int, int] = {
            f.floor_number: f.free_count() for f in floors
        }

    def on_spot_changed(
        self,
        spot:       ParkingSpot,
        new_status: SpotStatus,
    ) -> None:
        fn = spot.floor_number
        if new_status == SpotStatus.FREE:
            self._counts[fn] = self._counts.get(fn, 0) + 1
        elif new_status == SpotStatus.OCCUPIED:
            self._counts[fn] = max(0, self._counts.get(fn, 0) - 1)

    def get_count(self, floor_number: int) -> int:
        return self._counts.get(floor_number, 0)

    def display(self) -> None:
        print("\n── Display Board ──────────────────")
        for fn, count in sorted(self._counts.items()):
            bar = "█" * count + "░" * max(0, 5 - count)
            print(f"  Floor {fn}: {count:3d} free  [{bar}]")
        print("───────────────────────────────────\n")


class AvailabilityAlertObserver(SpotStatusObserver):
    """
    Fires an alert when a floor drops to or below `threshold` free spots.
    Adding this notification channel required zero changes to ParkingSpot,
    ParkingFloor, or ParkingLot — just register at startup.
    """

    def __init__(self, threshold: int = 2):
        self._threshold = threshold
        self._counts:    dict[int, int] = {}

    def on_spot_changed(
        self,
        spot:       ParkingSpot,
        new_status: SpotStatus,
    ) -> None:
        fn = spot.floor_number
        if new_status == SpotStatus.FREE:
            self._counts[fn] = self._counts.get(fn, 0) + 1
        elif new_status == SpotStatus.OCCUPIED:
            prev = self._counts.get(fn, 0)
            self._counts[fn] = max(0, prev - 1)
            if self._counts[fn] <= self._threshold:
                print(f"  ⚠  ALERT: Floor {fn} has only "
                      f"{self._counts[fn]} free spot(s) left!")


# ─────────────────────────────────────────────
# 10. PARKING LOT  (Singleton + Facade)
# ─────────────────────────────────────────────

class ParkingLot:
    """
    The Facade / Orchestrator.
    All external code talks only to this class.

    Singleton: one physical lot → one instance.
    Double-checked locking: avoids lock acquisition on every get_instance()
    call once the instance is created.

    The _lock inside the instance protects spot assignment.
    Lock scope is kept minimal: only the find-and-assign step is
    inside the lock. Ticket creation, fee calculation, and dict
    registration happen outside — they don't touch shared spot state.
    """

    _instance:      Optional[ParkingLot] = None
    _class_lock:    threading.Lock       = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        name:           str                   = "Default Lot",
        floors:         Optional[list[ParkingFloor]] = None,
        strategy:       Optional[ParkingStrategy]    = None,
        fee_calculator: Optional[FeeCalculator]      = None,
    ) -> ParkingLot:
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls(
                        name, floors or [], strategy, fee_calculator
                    )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """For testing only — clears the singleton so tests are isolated."""
        with cls._class_lock:
            cls._instance = None

    # ── init (private — use get_instance()) ──

    def __init__(
        self,
        name:           str,
        floors:         list[ParkingFloor],
        strategy:       Optional[ParkingStrategy],
        fee_calculator: Optional[FeeCalculator],
    ):
        self.name           = name
        self.floors         = floors
        self.strategy       = strategy       or NearestSpotStrategy()
        self.fee_calculator = fee_calculator or HourlyFeeCalculator()

        # active_tickets keyed by license_plate → O(1) lookup at exit.
        # Ticket objects survive vehicle exit as audit records.
        self._active_tickets: dict[str, Ticket] = {}
        self._all_tickets:    list[Ticket]      = []  # history
        self._lock:           threading.Lock    = threading.Lock()

    # ── core operations ──

    def park_vehicle(self, vehicle: Vehicle) -> Ticket:
        """
        Step 1  Acquire lock           — prevents race on spot assignment.
        Step 2  Check double-park       — fail fast before running strategy.
        Step 3  Delegate to strategy    — ParkingLot never searches itself.
        Step 4  Spot assigns itself     — Tell Don't Ask; spot owns its rules.
        Step 5  Release lock            — done with shared state.
        Step 6  Create Ticket           — pure in-memory work, no lock needed.
        Step 7  Register ticket         — dict.update is fast; outside lock.
        """
        with self._lock:
            if vehicle.license_plate in self._active_tickets:
                raise AlreadyParkedError(vehicle.license_plate)

            spot = self.strategy.find_spot(self.floors, vehicle)
            if spot is None:
                raise LotFullError()

            spot.assign_vehicle(vehicle)   # spot enforces its own invariant

        # Outside the lock — Ticket is a new object with no shared state.
        ticket = Ticket(vehicle=vehicle, spot=spot)
        self._active_tickets[vehicle.license_plate] = ticket
        self._all_tickets.append(ticket)
        return ticket

    def exit_vehicle(self, license_plate: str) -> float:
        """
        Step 1  pop() atomically removes the ticket.
                Second call for same plate gets None → safe double-exit guard.
        Step 2  Stamp exit_time BEFORE calculating fee so duration is exact.
        Step 3  Delegate fee calculation to fee_calculator.
        Step 4  Release spot under lock — another park_vehicle() must not
                see the spot as free before release is complete.
        """
        ticket = self._active_tickets.pop(license_plate, None)
        if ticket is None:
            raise TicketNotFoundError(license_plate)

        ticket.exit_time = datetime.now()             # freeze the timestamp
        ticket.fee       = self.fee_calculator.calculate(ticket)

        with self._lock:
            ticket.spot.release_vehicle()

        return ticket.fee

    # ── query helpers ──

    def get_availability(self) -> dict[int, int]:
        """Returns {floor_number: free_spot_count}."""
        return {f.floor_number: f.free_count() for f in self.floors}

    def get_active_ticket(self, license_plate: str) -> Optional[Ticket]:
        return self._active_tickets.get(license_plate)

    def ticket_history(self) -> list[Ticket]:
        """All tickets ever issued, including closed ones."""
        return list(self._all_tickets)

    def __repr__(self) -> str:
        total = sum(len(f.spots) for f in self.floors)
        free  = sum(f.free_count() for f in self.floors)
        return f"ParkingLot({self.name!r}, {free}/{total} free)"


# ─────────────────────────────────────────────
# 11. DEMO — end-to-end smoke test
# ─────────────────────────────────────────────

def build_sample_lot() -> ParkingLot:
    """
    Builds a 2-floor lot:
      Floor 1 — 3 small (motorcycle), 4 medium (car), 2 large (truck)
                1 medium HANDICAPPED spot, 1 medium EV_CHARGING spot
      Floor 2 — 3 small, 4 medium, 2 large
    Attaches DisplayBoard and Alert observers to every spot.
    """
    ParkingLot.reset()   # allow re-running demo in the same process

    def make_spots(
        floor_n: int,
        small: int = 3,
        medium: int = 4,
        large: int = 2,
    ) -> list[ParkingSpot]:
        spots: list[ParkingSpot] = []
        idx = 1
        for _ in range(small):
            spots.append(ParkingSpot(
                f"F{floor_n}-S{idx}", SpotSize.SMALL,
                SpotType.STANDARD, floor_n))
            idx += 1
        for _ in range(medium):
            spots.append(ParkingSpot(
                f"F{floor_n}-M{idx}", SpotSize.MEDIUM,
                SpotType.STANDARD, floor_n))
            idx += 1
        # one handicapped medium
        spots.append(ParkingSpot(
            f"F{floor_n}-H{idx}", SpotSize.MEDIUM,
            SpotType.HANDICAPPED, floor_n))
        idx += 1
        # one EV medium
        spots.append(ParkingSpot(
            f"F{floor_n}-E{idx}", SpotSize.MEDIUM,
            SpotType.EV_CHARGING, floor_n))
        idx += 1
        for _ in range(large):
            spots.append(ParkingSpot(
                f"F{floor_n}-L{idx}", SpotSize.LARGE,
                SpotType.STANDARD, floor_n))
            idx += 1
        return spots

    floor1_spots = make_spots(1)
    floor2_spots = make_spots(2)

    floor1 = ParkingFloor(1, floor1_spots)
    floor2 = ParkingFloor(2, floor2_spots)

    all_spots = floor1_spots + floor2_spots

    # Attach observers to every spot
    display = DisplayBoardObserver([floor1, floor2])
    alert   = AvailabilityAlertObserver(threshold=2)
    for spot in all_spots:
        spot.add_observer(display)
        spot.add_observer(alert)

    lot = ParkingLot.get_instance(
        name           = "Central Mall Parking",
        floors         = [floor1, floor2],
        strategy       = NearestSpotStrategy(),
        fee_calculator = HourlyFeeCalculator(),
    )
    return lot


def _sep(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


def run_demo() -> None:
    lot = build_sample_lot()

    _sep("Initial state")
    print(lot)
    print("Availability:", lot.get_availability())

    # ── park several vehicles ──
    _sep("Parking vehicles")

    bike   = VehicleFactory.create("KA01-BK-0001", "MOTORCYCLE")
    car1   = VehicleFactory.create("KA05-CA-1234", "CAR")
    car2   = VehicleFactory.create("MH12-CA-5678", "CAR")
    truck1 = VehicleFactory.create("TN22-TR-9999", "TRUCK")

    t_bike   = lot.park_vehicle(bike)
    t_car1   = lot.park_vehicle(car1)
    t_car2   = lot.park_vehicle(car2)
    t_truck1 = lot.park_vehicle(truck1)

    print(f"  Parked {bike}  → {t_bike}")
    print(f"  Parked {car1} → {t_car1}")
    print(f"  Parked {car2} → {t_car2}")
    print(f"  Parked {truck1} → {t_truck1}")

    _sep("State after parking 4 vehicles")
    print(lot)
    print("Availability:", lot.get_availability())

    # ── double-park guard ──
    _sep("Double-park guard")
    try:
        lot.park_vehicle(car1)
    except AlreadyParkedError as e:
        print(f"  ✓ Caught expected error: {e}")

    # ── exit one vehicle ──
    _sep("Exiting car1 (KA05-CA-1234)")
    # Simulate ~2.5 hours by overriding entry_time
    t_car1.entry_time = datetime(2024, 1, 1, 10, 0, 0)
    t_car1.exit_time  = None   # reset so exit_vehicle stamps it fresh

    # Because we manually adjusted entry_time, compute fee directly:
    t_car1.exit_time = datetime(2024, 1, 1, 12, 30, 0)  # 2h30m → ceil=3h
    t_car1.fee       = lot.fee_calculator.calculate(t_car1)
    with lot._lock:
        t_car1.spot.release_vehicle()
    del lot._active_tickets[car1.license_plate]   # manually close ticket

    print(f"  Fee for {car1}: ₹{t_car1.fee:.0f}")
    print(f"  (3 hours × ₹50/hr = ₹150 for CAR)")

    _sep("State after exit")
    print(lot)

    # ── park again in freed spot ──
    _sep("Parking new car in just-freed spot")
    car3 = VehicleFactory.create("DL01-CA-2222", "CAR")
    t_car3 = lot.park_vehicle(car3)
    print(f"  Parked {car3} → {t_car3}")

    # ── ticket-not-found guard ──
    _sep("Exit a vehicle that is not parked")
    try:
        lot.exit_vehicle("XX00-ZZ-0000")
    except TicketNotFoundError as e:
        print(f"  ✓ Caught expected error: {e}")

    # ── lot full scenario (fill remaining medium spots) ──
    _sep("Filling all medium spots, then attempt overflow")
    extra_plates = [f"KA99-XX-{i:04d}" for i in range(10)]
    parked_extra = []
    for plate in extra_plates:
        try:
            v = Car(plate)
            t = lot.park_vehicle(v)
            parked_extra.append(v)
        except LotFullError:
            print(f"  ✓ Lot full for {plate!r} — LotFullError raised correctly.")
            break

    # ── display board ──
    _sep("Display board (via observer)")
    # Retrieve the display observer from the first spot's observer list
    display_obs: Optional[DisplayBoardObserver] = None
    for spot in lot.floors[0].spots:
        for obs in spot._observers:
            if isinstance(obs, DisplayBoardObserver):
                display_obs = obs
                break
    if display_obs:
        display_obs.display()

    # ── strategy swap — HandicappedFirst ──
    _sep("Strategy swap: HandicappedFirstStrategy")
    lot.strategy = HandicappedFirstStrategy()
    car_hc = Car("KA03-HC-0001")
    # exit a parked car to free up a spot first
    if parked_extra:
        try:
            lot.exit_vehicle(parked_extra[0].license_plate)
        except Exception:
            pass
    try:
        t_hc = lot.park_vehicle(car_hc)
        print(f"  Parked {car_hc} at spot {t_hc.spot.spot_id!r}"
              f" (type: {t_hc.spot.spot_type.name})")
        if t_hc.spot.spot_type == SpotType.HANDICAPPED:
            print("  ✓ Correctly assigned to HANDICAPPED spot first.")
        else:
            print("  ℹ  No HANDICAPPED spot was free; assigned to standard spot.")
    except LotFullError:
        print("  ℹ  Lot full — no spot available even with HandicappedFirst.")

    # ── ticket history ──
    _sep("Ticket history (all issued tickets)")
    for t in lot.ticket_history():
        status = f"₹{t.fee:.0f}" if t.is_closed() else "active"
        print(f"  {t.ticket_id[:8]}…  {t.vehicle.license_plate:<16}"
              f"  spot={t.spot.spot_id:<8}  {status}")

    _sep("Demo complete")
    print(lot)


if __name__ == "__main__":
    run_demo()
