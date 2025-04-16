# netro/core/entities/route.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .location import Location
from .vehicle import Vehicle


@dataclass
class Route:
    """
    Represents a vehicle route.
    """

    vehicle: Vehicle
    locations: List[Location] = field(default_factory=list)
    start_time: float = 0.0

    @property
    def total_distance(self) -> float:
        """Calculate the total distance of the route."""
        if not self.locations:
            return 0.0

        total = 0.0
        for i in range(len(self.locations) - 1):
            total += self.locations[i].distance_to(self.locations[i + 1])
        return total

    @property
    def total_time(self) -> float:
        """Calculate the total time of the route."""
        return self.total_distance / self.vehicle.speed

    @property
    def total_cost(self) -> float:
        """Calculate the total cost of the route."""
        return (
            self.total_distance * self.vehicle.cost_per_distance
            + self.total_time * self.vehicle.cost_per_time
        )

    @property
    def total_emissions(self) -> float:
        """Calculate the total emissions of the route."""
        return self.total_distance * self.vehicle.emissions_per_distance

    @property
    def total_demand(self) -> float:
        """Calculate the total demand of the route."""
        return sum(loc.demand for loc in self.locations[1:-1])  # Exclude depot

    def is_feasible(self) -> bool:
        """Check if the route is feasible in terms of capacity."""
        return self.total_demand <= self.vehicle.capacity
