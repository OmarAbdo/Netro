"""
CustomerAnalyzer Module
-------------------------
This module implements the CustomerAnalyzer class to perform customer analysis on the Solomon dataset.
It extracts key information such as total demand, distribution of customer locations, and performs clustering using HDBSCAN.
HDBSCAN automatically determines the number of clusters based on the density of the data.

Inspiration and References:
    - McInnes, L., Healy, J., & Astels, S. (2017), "HDBSCAN: Hierarchical Density Based Clustering", The Journal of Open Source Software.
      (For robust, automatic cluster detection; see p. 2, lines 20–30.)
    - Solomon, M.M. (1987), "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints".
      (For basic dataset analysis and demand calculation.)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import sys
import os

sys.path.append(os.getcwd())
from utils.distance_matrix_calculator import DistanceMatrixCalculator


class CustomerAnalyzer:
    """
    CustomerAnalyzer loads and analyzes customer data from a Solomon dataset.
    It computes statistics such as total demand and clusters customers based on geographic coordinates.
    """

    def __init__(self, file_path):
        """
        Initialize the CustomerAnalyzer with the path to the Solomon dataset.

        :param file_path: Path to the Solomon VRPTW instance file.
        """
        self.file_path = file_path
        self.customers_df = None

    def load_data(self):
        """
        Load and parse the Solomon dataset using DistanceMatrixCalculator.
        Assumes the depot is the first row with ID == 0.
        """
        calculator = DistanceMatrixCalculator(self.file_path)
        calculator.load_data()
        self.customers_df = calculator.customers
        return self.customers_df

    def compute_statistics(self):
        """
        Compute basic statistics from the customer data.

        :return: Dictionary with total demand, number of customers (excluding depot), and average demand.
        """
        if self.customers_df is None:
            raise ValueError("Customer data not loaded. Call load_data() first.")
        df = self.customers_df[self.customers_df["ID"] != 0]
        total_demand = df["Demand"].sum()
        num_customers = len(df)
        avg_demand = df["Demand"].mean()
        return {
            "total_demand": total_demand,
            "num_customers": num_customers,
            "avg_demand": avg_demand,
        }

    def plot_customers(self):
        """
        Plot customer locations on a 2D scatter plot, highlighting the depot.
        """
        if self.customers_df is None:
            raise ValueError("Customer data not loaded. Call load_data() first.")
        plt.figure(figsize=(8, 8))
        depot = self.customers_df[self.customers_df["ID"] == 0]
        customers = self.customers_df[self.customers_df["ID"] != 0]
        plt.scatter(customers["X"], customers["Y"], c="blue", label="Customers")
        plt.scatter(depot["X"], depot["Y"], c="red", marker="s", label="Depot")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Customer Locations")
        plt.legend()
        plt.grid(True)
        plt.show()

    def cluster_customers_hdbscan(self, min_cluster_size=5, min_samples=None):
        """
        Cluster customers using HDBSCAN, which automatically determines the number of clusters.

        :param min_cluster_size: The minimum size of clusters.
        :param min_samples: The minimum samples in a neighborhood for a point to be a core point.
                            If None, defaults to min_cluster_size.
        :return: A tuple (labels, clusterer, n_clusters) where:
                 - labels: Array of cluster assignments for each customer (excluding depot).
                 - clusterer: The fitted HDBSCAN clusterer object.
                 - n_clusters: Number of clusters found (ignoring noise labeled as -1).

        Citation: Inspired by McInnes et al. (2017), "HDBSCAN: Hierarchical Density Based Clustering".
        """
        if self.customers_df is None:
            raise ValueError("Customer data not loaded. Call load_data() first.")
        df = self.customers_df[self.customers_df["ID"] != 0]
        coords = df[["X", "Y"]].values
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        )
        labels = clusterer.fit_predict(coords)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return labels, clusterer, n_clusters

    def compute_cluster_demands(self, labels):
        """
        Compute the total demand for each cluster based on the provided labels.

        :param labels: Array-like cluster labels for each customer (excluding depot).
        :return: Dictionary mapping cluster label to total demand.
        """
        if self.customers_df is None:
            raise ValueError("Customer data not loaded. Call load_data() first.")
        df = self.customers_df[self.customers_df["ID"] != 0].copy()
        df["Cluster"] = labels
        cluster_demands = df.groupby("Cluster")["Demand"].sum().to_dict()
        return cluster_demands

    @staticmethod
    def orders_from_solomon_df(customers_df):
        """
        Convert a Solomon dataset DataFrame into a list of orders.
        Assumes the depot is the first row (ID == 0) and skips it.

        :param customers_df: Pandas DataFrame with Solomon dataset columns including 'ID' and 'Demand'
        :return: List of order dictionaries.
        """
        orders_df = customers_df[customers_df["ID"] != 0]
        orders = []
        for _, row in orders_df.iterrows():
            orders.append({"id": int(row["ID"]), "demand": int(row["Demand"])})
        return orders

    def plot_clusters(self, labels):
        """
        Plot customer clusters using provided cluster labels.
        Noise points (label = -1) are plotted in black.

        :param labels: Array-like cluster labels for each customer (excluding depot).
        """
        if self.customers_df is None:
            raise ValueError("Customer data not loaded. Call load_data() first.")
        df = self.customers_df[self.customers_df["ID"] != 0].copy()
        df["Cluster"] = labels
        plt.figure(figsize=(8, 8))
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        for k, col in zip(unique_labels, colors):
            class_member_mask = df["Cluster"] == k
            if k == -1:
                col = [0, 0, 0, 1]  # Black for noise
                label_text = "Noise"
            else:
                label_text = f"Cluster {k}"
            xy = df[class_member_mask][["X", "Y"]]
            plt.scatter(
                xy["X"], xy["Y"], c=[col], label=label_text, edgecolor="k", s=50
            )
        plt.title("Customer Clusters (HDBSCAN)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.getcwd())

    file_path = "dataset/c101.txt"
    analyzer = CustomerAnalyzer(file_path)
    df = analyzer.load_data()
    print("Customer data loaded. Sample:")
    print(df.head())

    stats = analyzer.compute_statistics()
    print("\n--- Customer Statistics ---")
    print(f"Total Demand: {stats['total_demand']}")
    print(f"Number of Customers (excluding depot): {stats['num_customers']}")
    print(f"Average Demand per Customer: {stats['avg_demand']:.2f}")

    # Plot customer locations
    # analyzer.plot_customers()

    # Cluster customers using HDBSCAN
    labels, clusterer, n_clusters = analyzer.cluster_customers_hdbscan(
        min_cluster_size=5
    )
    print(f"\nHDBSCAN found {n_clusters} clusters (noise labeled as -1).")
    analyzer.plot_clusters(labels)

    # Compute and print total demand per cluster
    cluster_demands = analyzer.compute_cluster_demands(labels)
    print("\n--- Cluster Demands ---")
    for cluster, demand in cluster_demands.items():
        print(f"Cluster {cluster}: Total Demand = {demand}")
