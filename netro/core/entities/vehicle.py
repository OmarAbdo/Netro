# netro/core/entities/vehicle.py
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Vehicle:
    """
    Base class for vehicles (trucks and robots).
    """

    id: int
    capacity: float
    speed: float  # in distance units per time unit
    cost_per_distance: float
    cost_per_time: float
    emissions_per_distance: float


@dataclass
class Truck(Vehicle):
    """
    Represents a delivery truck that can carry robots.
    """

    robot_capacity: int  # Number of robots the truck can carry
    loading_time: float  # Time to load/unload a robot

    def calculate_trip_cost(self, distance: float, time: float) -> float:
        """Calculate the total cost of a trip."""
        return distance * self.cost_per_distance + time * self.cost_per_time

    def calculate_emissions(self, distance: float) -> float:
        """Calculate the emissions for a trip."""
        return distance * self.emissions_per_distance


@dataclass
class Robot(Vehicle):
    """
    Represents an autonomous delivery robot.
    """

    battery_capacity: float  # in time units
    recharging_rate: float  # time units per charge unit

    def can_complete_trip(self, distance: float) -> bool:
        """Check if the robot can complete a trip with the current battery."""
        trip_time = distance / self.speed
        return trip_time <= self.battery_capacity
