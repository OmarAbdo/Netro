# netro/services/clustering/hdbscan_clustering.py
import numpy as np
import hdbscan
from typing import Tuple, List, Dict, Optional, Set
from netro.core.interfaces.clustering import ClusteringAlgorithm


class HDBSCANClusterer:
    """
    Implements clustering using HDBSCAN algorithm with noise reassignment.

    Based on:
    McInnes, L., Healy, J., & Astels, S. (2017), "HDBSCAN: Hierarchical Density Based Clustering",
    The Journal of Open Source Software, p. 2, lines 20–30.
    """

    def __init__(self, min_cluster_size: int = 5, min_samples: Optional[int] = None):
        """
        Initialize HDBSCAN clustering algorithm.

        Args:
            min_cluster_size: The minimum size of clusters.
            min_samples: The minimum number of samples in a neighborhood for a point
                        to be considered a core point. Defaults to min_cluster_size.
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size

    def cluster(self, coordinates: np.ndarray, **kwargs) -> Tuple[np.ndarray, int]:
        """
        Cluster points based on their coordinates using HDBSCAN.

        Args:
            coordinates: An array of shape (n_samples, n_features) containing point coordinates.
            **kwargs: Additional parameters to pass to HDBSCAN.

        Returns:
            A tuple containing:
            - An array of cluster labels for each point.
            - The number of clusters found.
        """
        # Create and fit HDBSCAN clusterer
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            **kwargs
        )
        labels = clusterer.fit_predict(coordinates)

        # Reassign noise points to nearest cluster
        labels = self._reassign_noise(coordinates, labels)

        # Count unique clusters
        n_clusters = len(set(labels))

        return labels, n_clusters

    def _reassign_noise(
        self, coordinates: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        Reassign points labeled as noise (-1) to their nearest non-noise cluster.

        Args:
            coordinates: An array of shape (n_samples, n_features) containing point coordinates.
            labels: Current cluster labels with noise points labeled as -1.

        Returns:
            Updated array of cluster labels with noise points reassigned.
        """
        # Make a copy of labels to modify
        updated_labels = labels.copy()

        # Find indices of noise points
        noise_indices = np.where(labels == -1)[0]

        # If there are no noise points or all points are noise, return original labels
        if len(noise_indices) == 0 or len(noise_indices) == len(labels):
            return labels

        # Get coordinates of non-noise points
        valid_indices = np.where(labels != -1)[0]
        valid_coords = coordinates[valid_indices]
        valid_labels = labels[valid_indices]

        # Compute centroids for each cluster
        unique_labels = np.unique(valid_labels)
        centroids = {}
        for label in unique_labels:
            centroids[label] = valid_coords[valid_labels == label].mean(axis=0)

        # For each noise point, assign to nearest centroid
        for idx in noise_indices:
            point = coordinates[idx]
            min_dist = float("inf")
            nearest_label = None

            for label, centroid in centroids.items():
                dist = np.linalg.norm(point - centroid)
                if dist < min_dist:
                    min_dist = dist
                    nearest_label = label

            if nearest_label is not None:
                updated_labels[idx] = nearest_label
            else:
                # If something goes wrong, just assign to the first valid cluster
                updated_labels[idx] = unique_labels[0]

        return updated_labels
