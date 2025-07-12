# netro/services/clustering/capacitated_splitter.py
import numpy as np
from typing import Tuple, Dict, List, Set
from netro.core.interfaces.clustering import CapacitatedSplitter


class CapacitatedClusterSplitter:
    """
    Splits clusters whose total demand exceeds capacity constraints.

    Based on:
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
        # Get unique clusters
        unique_labels = np.unique(labels)

        # Keep track of new labels and centroids
        new_labels = np.full_like(labels, -1)
        cluster_demands = {}
        next_label = max(unique_labels) + 1
        centroids_list = []

        # Process each original cluster
        for label in unique_labels:
            # Get indices of points in this cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_coords = coordinates[cluster_indices]
            cluster_demands_arr = demands[cluster_indices]

            # Check if cluster needs splitting
            total_demand = np.sum(cluster_demands_arr)

            if total_demand <= capacity:
                # Cluster fits within capacity, no splitting needed
                new_labels[cluster_indices] = label
                cluster_demands[label] = float(total_demand)
                centroids_list.append(np.mean(cluster_coords, axis=0))
            else:
                # Sort points by demand (descending)
                sorted_indices = np.argsort(-cluster_demands_arr)

                # Track which subcluster each point will go into
                subcluster_assignments = {}
                subcluster_current_demand = {}

                # Assign points to subclusters using a greedy bin-packing approach
                for i in sorted_indices:
                    point_idx = cluster_indices[i]
                    point_demand = demands[point_idx]

                    # Try to find an existing subcluster with enough capacity
                    assigned = False
                    for (
                        subcluster_label,
                        current_demand,
                    ) in subcluster_current_demand.items():
                        if current_demand + point_demand <= capacity:
                            subcluster_assignments[point_idx] = subcluster_label
                            subcluster_current_demand[subcluster_label] += point_demand
                            assigned = True
                            break

                    # If can't fit in existing subclusters, create a new one
                    if not assigned:
                        new_label_id = next_label
                        next_label += 1
                        subcluster_assignments[point_idx] = new_label_id
                        subcluster_current_demand[new_label_id] = point_demand

                # Update new_labels and cluster_demands with subclusters
                for point_idx, subcluster_label in subcluster_assignments.items():
                    new_labels[point_idx] = subcluster_label

                for subcluster_label, demand in subcluster_current_demand.items():
                    cluster_demands[subcluster_label] = float(demand)

                    # Calculate centroid for this subcluster
                    subcluster_points = coordinates[
                        [
                            idx
                            for idx, label in subcluster_assignments.items()
                            if label == subcluster_label
                        ]
                    ]
                    centroids_list.append(np.mean(subcluster_points, axis=0))

        # Convert centroids list to numpy array
        centroids = np.array(centroids_list)

        return new_labels, centroids, cluster_demands
