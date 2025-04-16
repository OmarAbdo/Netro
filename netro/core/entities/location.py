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
