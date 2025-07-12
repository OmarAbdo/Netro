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
