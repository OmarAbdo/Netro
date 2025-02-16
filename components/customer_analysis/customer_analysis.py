"""
CustomerAnalyzer Module
-------------------------
This module implements the CustomerAnalyzer class to perform customer analysis on the Solomon dataset.
It extracts key information such as total demand, distribution of customer locations, and performs clustering using K-means.

Inspiration and References:
    - Solomon, M.M. (1987), "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints".
      (For basic dataset analysis and demand calculation. See page 3, lines 10-15 for demand summary approaches.)
    - Mourelo Ferrandez et al. (2016), "Optimization of a Truck-drone in Tandem Delivery Network Using K-means and Genetic Algorithm".
      (For clustering ideas using K-means; see page 377, lines 10-15.)
      
This module ignores time window constraints as requested.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys
import os

sys.path.append(os.getcwd())
from utils.distance_matrix_calculator import (
    DistanceMatrixCalculator,
)  # assuming this is in utils folder


class CustomerAnalyzer:
    """
    CustomerAnalyzer class loads and analyzes customer data from a Solomon dataset.
    It computes statistics such as total demand and clusters customers based on their geographic coordinates.
    """

    def __init__(self, file_path):
        """
        Initialize the CustomerAnalyzer with the path to the Solomon dataset.

        :param file_path: Path to the Solomon VRPTW instance file.
        """
        self.file_path = file_path
        self.customers_df = None  # DataFrame holding customer data

    def load_data(self):
        """
        Load and parse the Solomon dataset.
        Uses DistanceMatrixCalculator to load data.

        Note: Assumes the depot is the first row with ID==0.
        """
        calculator = DistanceMatrixCalculator(self.file_path)
        calculator.load_data()
        self.customers_df = calculator.customers
        return self.customers_df

    def compute_statistics(self):
        """
        Compute and return basic statistics from the customer data.

        :return: A dictionary with total demand, number of customers (excluding depot), average demand.
        """
        if self.customers_df is None:
            raise ValueError("Customer data not loaded. Call load_data() first.")

        # Exclude depot (assumed to have ID 0)
        df = self.customers_df[self.customers_df["ID"] != 0]
        total_demand = df["Demand"].sum()
        num_customers = len(df)
        avg_demand = df["Demand"].mean()

        stats = {
            "total_demand": total_demand,
            "num_customers": num_customers,
            "avg_demand": avg_demand,
        }
        return stats

    def plot_customers(self):
        """
        Plot customer locations on a 2D scatter plot.
        The depot is highlighted in red and other customers in blue.
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

    def cluster_customers(self, n_clusters):
        """
        Cluster customers based on their (X, Y) coordinates using K-means.

        :param n_clusters: Number of clusters to form.
        :return: A tuple (labels, centroids), where 'labels' is an array of cluster assignments
                 for each customer (excluding depot) and 'centroids' is the array of cluster centers.

        Citation: This approach is inspired by Mourelo Ferrandez et al. (2016), p. 377, lines 10-15.
        """
        if self.customers_df is None:
            raise ValueError("Customer data not loaded. Call load_data() first.")

        # Exclude depot for clustering
        df = self.customers_df[self.customers_df["ID"] != 0]
        coords = df[["X", "Y"]].values

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(coords)
        centroids = kmeans.cluster_centers_
        return labels, centroids


if __name__ == "__main__":
    # Usage Example for CustomerAnalyzer
    file_path = "dataset/c101.txt"  # Path to the Solomon dataset

    analyzer = CustomerAnalyzer(file_path)

    # Load data
    df = analyzer.load_data()
    print("Customer data loaded. Sample:")
    print(df.head())

    # Compute statistics
    stats = analyzer.compute_statistics()
    print("\n--- Customer Statistics ---")
    print(f"Total Demand: {stats['total_demand']}")
    print(f"Number of Customers (excluding depot): {stats['num_customers']}")
    print(f"Average Demand per Customer: {stats['avg_demand']:.2f}")

    # Plot customer locations
    analyzer.plot_customers()

    # Cluster customers into, say, 5 clusters
    n_clusters = 5
    labels, centroids = analyzer.cluster_customers(n_clusters)
    print("\nCluster Labels for Customers:")
    print(labels)
    print("\nCluster Centroids (X, Y):")
    print(centroids)

    # Optionally, plot clustering results
    import matplotlib.pyplot as plt

    df_customers = df.copy()
    df_customers["Cluster"] = labels
    plt.figure(figsize=(8, 8))
    for i in range(n_clusters):
        cluster_points = df_customers[df_customers["Cluster"] == i]
        plt.scatter(cluster_points["X"], cluster_points["Y"], label=f"Cluster {i}")
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        marker="x",
        s=100,
        label="Centroids",
    )
    plt.title("Customer Clusters")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()
