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
