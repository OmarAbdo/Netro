# netro/services/clustering/cluster_service.py
from typing import List, Dict, Tuple, Optional
import numpy as np
from netro.core.entities.location import Location
from netro.core.entities.cluster import Cluster
from netro.core.interfaces.clustering import ClusteringAlgorithm, CapacitatedSplitter


class ClusterService:
    """
    High-level service for clustering operations.
    Orchestrates the clustering and capacitated splitting process.
    """

    def __init__(
        self,
        clustering_algorithm: ClusteringAlgorithm,
        capacitated_splitter: CapacitatedSplitter,
    ):
        """
        Initialize the cluster service.

        Args:
            clustering_algorithm: An implementation of the ClusteringAlgorithm protocol.
            capacitated_splitter: An implementation of the CapacitatedSplitter protocol.
        """
        self.clustering_algorithm = clustering_algorithm
        self.capacitated_splitter = capacitated_splitter

    def cluster_locations(
        self, locations: List[Location], truck_capacity: float, **clustering_params
    ) -> Tuple[List[Cluster], Dict[int, Location]]:
        """
        Cluster locations and ensure each cluster respects capacity constraints.

        Args:
            locations: List of customer locations to cluster.
            truck_capacity: Maximum capacity of a truck.
            **clustering_params: Additional parameters to pass to the clustering algorithm.

        Returns:
            A tuple containing:
            - A list of Cluster objects.
            - A dictionary mapping cluster IDs to centroid locations.
        """
        # Extract coordinates and demands
        coordinates = np.array([loc.coordinates() for loc in locations])
        demands = np.array([loc.demand for loc in locations])

        # Perform initial clustering
        labels, n_clusters = self.clustering_algorithm.cluster(
            coordinates, **clustering_params
        )

        # Split clusters to respect capacity
        split_labels, centroids, cluster_demands = self.capacitated_splitter.split(
            coordinates, demands, labels, truck_capacity
        )

        # Create Cluster objects
        clusters = {}
        for i, label in enumerate(split_labels):
            if label not in clusters:
                clusters[label] = Cluster(id=int(label), locations=[])
            clusters[label].add_location(locations[i])

        # Create centroid locations
        centroid_locations = {}
        for i, (label, cluster) in enumerate(clusters.items()):
            centroid_loc = Location(
                id=-int(label) - 1000,  # Use negative ID to avoid conflicts
                x=float(cluster.centroid[0]),
                y=float(cluster.centroid[1]),
                demand=float(cluster_demands[label]),
            )
            centroid_locations[label] = centroid_loc

        return list(clusters.values()), centroid_locations
