"""
CapacitatedClusterSplitter Module
------------------------------------
This module takes initial cluster labels obtained from HDBSCAN (via CustomerAnalyzer.cluster_customers_hdbscan)
and splits clusters whose total demand exceeds the truck capacity. Noise points (labeled as -1 by HDBSCAN) are
treated as individual clusters.

Inspiration and References:
    - McInnes, L., Healy, J., & Astels, S. (2017), "HDBSCAN: Hierarchical Density Based Clustering", The Journal of Open Source Software,
      p. 2, lines 20–30.
    - Mourelo Ferrandez et al. (2016), "Optimization of a Truck-drone in Tandem Delivery Network Using K-means and Genetic Algorithm",
      JIEM, p. 377, lines 10–15.
    - Concepts from capacitated clustering literature (e.g., Beasley, 1990).

This module follows SOLID principles with one class per file.
"""

import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import pandas as pd


class CapacitatedClusterSplitter:
    """
    CapacitatedClusterSplitter processes initial clusters (from HDBSCAN) and splits any cluster
    whose total demand exceeds the truck capacity. Noise points (with label -1) are treated individually.
    """

    def __init__(self, customers_df, initial_labels, truck_capacity):
        """
        :param customers_df: DataFrame containing customer data (assumes depot already excluded)
        :param initial_labels: Array-like cluster labels from HDBSCAN.
        :param truck_capacity: Maximum capacity of a truck.
        """
        self.customers_df = customers_df.copy()
        self.customers_df["Cluster"] = initial_labels
        self.truck_capacity = truck_capacity

    def split_clusters(self):
        """
        For each cluster (ignoring noise, which is labeled as -1), check if the total demand exceeds truck capacity.
        If so, split the cluster into subclusters so that each subcluster's demand is <= truck_capacity.
        Noise points (label -1) are kept as separate clusters.

        :return: A tuple (new_labels, new_centroids, subcluster_mapping, new_cluster_demands) where:
                 - new_labels: New cluster labels for each customer.
                 - new_centroids: Array of centroids for the new clusters.
                 - subcluster_mapping: Dictionary mapping new cluster labels to the original cluster label.
                 - new_cluster_demands: Dictionary mapping new cluster labels to the total demand of that subcluster.

        Citation: This approach is inspired by McInnes et al. (2017) and capacitated clustering ideas in Mourelo Ferrandez et al. (2016).
        """
        new_labels = []
        subcluster_mapping = {}  # Maps new label -> original cluster label
        new_cluster_demands = {}  # Maps new label -> total demand in that subcluster
        new_centroids_list = []
        new_label_counter = 0

        unique_clusters = sorted(self.customers_df["Cluster"].unique())
        for cluster in unique_clusters:
            if cluster == -1:
                # Treat each noise point as its own cluster
                noise_df = self.customers_df[self.customers_df["Cluster"] == -1]
                for idx, row in noise_df.iterrows():
                    new_labels.append(new_label_counter)
                    subcluster_mapping[new_label_counter] = -1
                    new_cluster_demands[new_label_counter] = row["Demand"]
                    new_centroids_list.append(np.array([row["X"], row["Y"]]))
                    new_label_counter += 1
                continue

            # Process valid clusters (cluster != -1)
            cluster_df = self.customers_df[self.customers_df["Cluster"] == cluster]
            total_demand = cluster_df["Demand"].sum()

            if total_demand <= self.truck_capacity:
                # Cluster is feasible; assign one new label for all customers in this cluster
                count = len(cluster_df)
                new_labels.extend([new_label_counter] * count)
                subcluster_mapping[new_label_counter] = cluster
                new_cluster_demands[new_label_counter] = total_demand
                centroid = cluster_df[["X", "Y"]].mean().values
                new_centroids_list.append(centroid)
                new_label_counter += 1
            else:
                # Split the cluster: sort orders by demand descending and greedily partition them
                sorted_df = cluster_df.sort_values(by="Demand", ascending=False).copy()
                temp_labels = [-1] * len(sorted_df)
                # Dictionary to track current demand in each subcluster for this original cluster
                current_subcluster_demand = {}
                for idx in sorted_df.index:
                    order_demand = sorted_df.loc[idx, "Demand"]
                    assigned = False
                    # Try to assign to an existing subcluster for this original cluster
                    for sub_label in [
                        label
                        for label, orig in subcluster_mapping.items()
                        if orig == cluster
                    ]:
                        if (
                            current_subcluster_demand.get(sub_label, 0) + order_demand
                            <= self.truck_capacity
                        ):
                            temp_labels[sorted_df.index.get_loc(idx)] = sub_label
                            current_subcluster_demand[sub_label] = (
                                current_subcluster_demand.get(sub_label, 0)
                                + order_demand
                            )
                            assigned = True
                            break
                    if not assigned:
                        # Create a new subcluster for this original cluster
                        temp_labels[sorted_df.index.get_loc(idx)] = new_label_counter
                        subcluster_mapping[new_label_counter] = cluster
                        current_subcluster_demand[new_label_counter] = order_demand
                        new_label_counter += 1
                new_labels.extend(temp_labels)
                # For each new subcluster created for this original cluster, compute centroid and record total demand
                unique_sub_labels = sorted(set(temp_labels))
                for sub_label in unique_sub_labels:
                    sub_df = sorted_df[
                        [temp_labels[i] == sub_label for i in range(len(sorted_df))]
                    ]
                    centroid = sub_df[["X", "Y"]].mean().values
                    new_centroids_list.append(centroid)
                    new_cluster_demands[sub_label] = current_subcluster_demand[
                        sub_label
                    ]

        new_labels = np.array(new_labels)
        new_centroids = np.array(new_centroids_list)
        return new_labels, new_centroids, subcluster_mapping, new_cluster_demands


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.getcwd())
    import matplotlib.pyplot as plt
    from components.customer_clustering.customer_analysis import CustomerAnalyzer
    import hdbscan

    # Path to Solomon dataset file
    file_path = "dataset/c101.txt"

    # Create and load customer data using CustomerAnalyzer
    analyzer = CustomerAnalyzer(file_path)
    df = analyzer.load_data()

    # Exclude depot (ID == 0)
    customers_df = df[df["ID"] != 0].copy()

    # Perform HDBSCAN clustering on customers
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    initial_labels = clusterer.fit_predict(customers_df[["X", "Y"]].values)
    n_clusters = len(set(initial_labels)) - (1 if -1 in initial_labels else 0)
    print(f"HDBSCAN initially found {n_clusters} clusters (noise labeled as -1).")

    # Define truck capacity (for example, 200 units)
    truck_capacity = 200

    # Initialize CapacitatedClusterSplitter with HDBSCAN labels
    splitter = CapacitatedClusterSplitter(customers_df, initial_labels, truck_capacity)
    new_labels, new_centroids, mapping, new_cluster_demands = splitter.split_clusters()

    print("New Cluster Labels:")
    print(new_labels)
    print("New Cluster Centroids:")
    print(new_centroids)
    print("Subcluster Mapping (new_label -> original cluster):")
    print(mapping)
    print("New Cluster Demands (Total demand per new cluster):")
    print(new_cluster_demands)

    # Plot new clusters
    customers_df["NewCluster"] = new_labels
    plt.figure(figsize=(8, 8))
    for label in np.unique(new_labels):
        cluster_points = customers_df[customers_df["NewCluster"] == label]
        plt.scatter(
            cluster_points["X"],
            cluster_points["Y"],
            label=f"Cluster {label}",
            edgecolor="k",
            s=50,
        )
    plt.scatter(
        new_centroids[:, 0],
        new_centroids[:, 1],
        c="black",
        marker="x",
        s=100,
        label="Centroids",
    )
    plt.title("Capacitated Customer Clusters (HDBSCAN based)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()
