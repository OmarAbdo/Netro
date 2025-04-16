# netro/services/benchmarking/comparison_service.py
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netro.core.entities.location import Location
from netro.core.entities.vehicle import Truck, Robot


class ComparisonService:
    """
    Service for comparing different delivery approaches:
    - Traditional truck-only delivery
    - Netro hybrid truck-robot delivery

    Provides detailed metrics and visualizations for comparison.
    """

    def compare(
        self, baseline_solution: Dict[str, Any], netro_solution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare baseline and Netro solutions.

        Args:
            baseline_solution: Solution from BaselineTruckService.
            netro_solution: Solution from NetroRoutingService.

        Returns:
            Dictionary with comparison metrics.
        """
        # Extract key metrics
        baseline_time = baseline_solution["total_time"]
        baseline_distance = baseline_solution["total_distance"]
        baseline_cost = baseline_solution["total_cost"]
        baseline_emissions = baseline_solution["total_emissions"]

        netro_time = netro_solution["total_time"]
        netro_truck_distance = netro_solution["total_truck_distance"]
        netro_robot_distance = netro_solution["total_robot_distance"]
        netro_total_distance = netro_truck_distance + netro_robot_distance

        # Compute cost and emissions for Netro (from truck_metrics and cluster_metrics)
        truck_metrics = netro_solution["truck_metrics"]
        cluster_metrics = netro_solution["cluster_metrics"]

        # Calculate improvement percentages
        time_improvement = (baseline_time - netro_time) / baseline_time * 100
        distance_improvement = (
            (baseline_distance - netro_total_distance) / baseline_distance * 100
        )

        # Create comparison dictionary
        comparison = {
            "baseline": {
                "total_time": baseline_time,
                "total_distance": baseline_distance,
                "total_cost": baseline_cost,
                "total_emissions": baseline_emissions,
                "num_trucks_used": baseline_solution["metrics"]["num_trucks_used"],
            },
            "netro": {
                "total_time": netro_time,
                "total_truck_distance": netro_truck_distance,
                "total_robot_distance": netro_robot_distance,
                "total_distance": netro_total_distance,
                "num_trucks_used": len(netro_solution["truck_routes"]),
                "num_clusters": len(netro_solution["cluster_routes"]),
            },
            "improvement": {
                "time_percent": time_improvement,
                "distance_percent": distance_improvement,
            },
        }

        return comparison

    def generate_report(self, comparison: Dict[str, Any]) -> str:
        """
        Generate a text report from comparison data.

        Args:
            comparison: Comparison data from the compare method.

        Returns:
            Text report as a string.
        """
        baseline = comparison["baseline"]
        netro = comparison["netro"]
        improvement = comparison["improvement"]

        report = []
        report.append("# Netro vs Traditional Truck-Only Delivery Comparison Report")
        report.append("")
        report.append("## Key Metrics")
        report.append("")
        report.append("| Metric | Traditional | Netro | Improvement |")
        report.append("|--------|------------|-------|-------------|")
        report.append(
            f"| Total Time (hours) | {baseline['total_time']:.2f} | {netro['total_time']:.2f} | {improvement['time_percent']:.2f}% |"
        )
        report.append(
            f"| Total Distance (km) | {baseline['total_distance']:.2f} | {netro['total_distance']:.2f} | {improvement['distance_percent']:.2f}% |"
        )
        report.append(
            f"| Number of Trucks | {baseline['num_trucks_used']} | {netro['num_trucks_used']} | - |"
        )
        report.append("")
        report.append("## Netro Additional Metrics")
        report.append("")
        report.append(f"- Number of Clusters: {netro['num_clusters']}")
        report.append(f"- Truck Distance: {netro['total_truck_distance']:.2f} km")
        report.append(f"- Robot Distance: {netro['total_robot_distance']:.2f} km")

        return "\n".join(report)

    def plot_comparison(self, comparison: Dict[str, Any]) -> plt.Figure:
        """
        Generate a bar chart comparing key metrics.

        Args:
            comparison: Comparison data from the compare method.

        Returns:
            Matplotlib figure with comparison plots.
        """
        baseline = comparison["baseline"]
        netro = comparison["netro"]

        metrics = ["Total Time", "Total Distance"]
        traditional_values = [baseline["total_time"], baseline["total_distance"]]
        netro_values = [netro["total_time"], netro["total_distance"]]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(x - width / 2, traditional_values, width, label="Traditional Truck-Only")
        ax.bar(x + width / 2, netro_values, width, label="Netro Hybrid")

        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Value")
        ax.set_title("Comparison of Traditional vs Netro Delivery")
        ax.legend()

        return fig
