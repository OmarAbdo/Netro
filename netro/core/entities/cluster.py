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
