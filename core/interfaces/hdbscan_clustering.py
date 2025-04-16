# netro/core/interfaces/clustering.py
from typing import Protocol, List, Dict, Tuple, Any, TypeVar, Generic
import numpy as np
from dataclasses import dataclass

T = TypeVar("T")


class ClusteringAlgorithm(Protocol):
    """
    Protocol defining clustering algorithm behavior.

    Inspired by:
    McInnes, L., Healy, J., & Astels, S. (2017), "HDBSCAN: Hierarchical Density Based Clustering",
    The Journal of Open Source Software, p. 2, lines 20–30.
    """

    def cluster(self, coordinates: np.ndarray, **kwargs) -> Tuple[np.ndarray, int]:
        """
        Cluster points based on their coordinates.

        Args:
            coordinates: An array of shape (n_samples, n_features) containing point coordinates.
            **kwargs: Additional algorithm-specific parameters.

        Returns:
            A tuple containing:
            - An array of cluster labels for each point.
            - The number of clusters found.
        """
        ...


class CapacitatedSplitter(Protocol):
    """
    Protocol for splitting clusters to ensure capacity constraints.

    Inspired by:
    Mourelo Ferrandez et al. (2016), "Optimization of a Truck-drone in Tandem Delivery Network
    Using K-means and Genetic Algorithm", JIEM, p. 377, lines 10–15.
    """

    def split(
        self,
        coordinates: np.ndarray,
        demands: np.ndarray,
        labels: np.ndarray,
        capacity: float,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, float]]:
        """
        Split clusters to ensure capacity constraints.

        Args:
            coordinates: An array of shape (n_samples, n_features) containing point coordinates.
            demands: An array of demand values for each point.
            labels: Initial cluster labels for each point.
            capacity: Maximum capacity per cluster.

        Returns:
            A tuple containing:
            - An array of updated cluster labels.
            - An array of cluster centroids.
            - A dictionary mapping cluster labels to total demand.
        """
        ...


# netro/core/interfaces/routing.py
from typing import Protocol, List, Dict, Tuple, Any
import numpy as np


class RoutingAlgorithm(Protocol):
    """
    Protocol for routing algorithms.

    Inspired by:
    Solomon, M.M. (1987), "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints".
    Chen, S., et al. (2018), "An adaptive large neighborhood search heuristic for dynamic vehicle routing problems".
    """

    def solve(
        self,
        distance_matrix: np.ndarray,
        demands: np.ndarray,
        capacities: np.ndarray,
        **kwargs
    ) -> Tuple[List[List[int]], float]:
        """
        Solve a capacitated vehicle routing problem.

        Args:
            distance_matrix: Matrix of distances between locations.
            demands: Array of demands for each location (index 0 is typically depot).
            capacities: Array of vehicle capacities.
            **kwargs: Additional algorithm-specific parameters.

        Returns:
            A tuple containing:
            - A list of routes, where each route is a list of location indices.
            - The total distance of all routes.
        """
        ...


class LocalSearch(Protocol):
    """
    Protocol for local search optimization.

    Inspired by:
    Chen, S., et al. (2018), "An adaptive large neighborhood search heuristic for
    dynamic vehicle routing problems", Computers and Electrical Engineering, 67, 596–607.
    """

    def improve(
        self,
        routes: List[List[int]],
        distance_matrix: np.ndarray,
        demands: np.ndarray,
        capacities: np.ndarray,
        **kwargs
    ) -> Tuple[List[List[int]], float]:
        """
        Improve existing routes using local search techniques.

        Args:
            routes: List of routes, where each route is a list of location indices.
            distance_matrix: Matrix of distances between locations.
            demands: Array of demands for each location.
            capacities: Array of vehicle capacities.
            **kwargs: Additional algorithm-specific parameters.

        Returns:
            A tuple containing:
            - Improved routes.
            - The new total distance.
        """
        ...


# netro/core/entities/location.py
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class Location:
    """
    Immutable value object representing a customer or depot location.
    """

    id: int
    x: float
    y: float
    demand: float
    ready_time: Optional[float] = None
    due_time: Optional[float] = None
    service_time: Optional[float] = None

    def distance_to(self, other: "Location") -> float:
        """Calculate Euclidean distance to another location."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def coordinates(self) -> np.ndarray:
        """Return coordinates as a NumPy array."""
        return np.array([self.x, self.y])


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


# netro/core/entities/cluster.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
import numpy as np
from .location import Location


@dataclass
class Cluster:
    """
    Represents a cluster of customer locations.
    """

    id: int
    locations: List[Location] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None

    def __post_init__(self):
        if not self.centroid and self.locations:
            self.update_centroid()

    def update_centroid(self) -> None:
        """Update the centroid based on current locations."""
        if not self.locations:
            return

        coords = np.array([loc.coordinates() for loc in self.locations])
        self.centroid = coords.mean(axis=0)

    def add_location(self, location: Location) -> None:
        """Add a location to the cluster and update the centroid."""
        self.locations.append(location)
        self.update_centroid()

    def remove_location(self, location: Location) -> None:
        """Remove a location from the cluster and update the centroid."""
        self.locations = [loc for loc in self.locations if loc.id != location.id]
        self.update_centroid()

    @property
    def total_demand(self) -> float:
        """Calculate the total demand of the cluster."""
        return sum(loc.demand for loc in self.locations)

    def to_array(self) -> np.ndarray:
        """Convert locations to a NumPy array of coordinates."""
        return np.array([loc.coordinates() for loc in self.locations])

    def get_centroid_location(self) -> Location:
        """Create a location representing the centroid."""
        if self.centroid is None:
            self.update_centroid()

        if self.centroid is None:  # Still None, no locations
            raise ValueError("Cannot create centroid location for empty cluster")

        return Location(
            id=-self.id - 1000,  # Negative ID to distinguish from real locations
            x=float(self.centroid[0]),
            y=float(self.centroid[1]),
            demand=self.total_demand,
        )
