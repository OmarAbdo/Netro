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
        baseline_cost = baseline_solution.get("total_cost", 0)
        baseline_emissions = baseline_solution.get("total_emissions", 0)

        # Use the parallel time calculation for Netro, not sequential
        netro_time = (
            netro_solution["parallel_time"]
            if "parallel_time" in netro_solution
            else netro_solution["total_time"]
        )
        netro_truck_distance = netro_solution["total_truck_distance"]
        netro_robot_distance = netro_solution["total_robot_distance"]
        netro_total_distance = netro_truck_distance + netro_robot_distance

        # Compute cost and emissions for Netro
        # This is simplified; in a complete implementation, we would get this from netro_solution
        netro_cost = (
            baseline_cost * (netro_time / baseline_time) if baseline_time > 0 else 0
        )
        netro_emissions = (
            baseline_emissions * (netro_truck_distance / baseline_distance)
            if baseline_distance > 0
            else 0
        )

        # Calculate improvement percentages
        time_improvement = (
            (baseline_time - netro_time) / baseline_time * 100
            if baseline_time > 0
            else 0
        )
        distance_improvement = (
            (baseline_distance - netro_total_distance) / baseline_distance * 100
            if baseline_distance > 0
            else 0
        )

        # Note: Distance "improvement" might be negative because robots travel more distance
        # but this is expected and offset by time savings and operational efficiency

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
                "estimated_cost": netro_cost,
                "estimated_emissions": netro_emissions,
                "num_trucks_used": len(netro_solution["truck_routes"]),
                "num_clusters": len(netro_solution["cluster_routes"]),
                "parallel_computation": True,  # Flag indicating we're using parallel time
            },
            "improvement": {
                "time_percent": time_improvement,
                "distance_percent": distance_improvement,
                "notes": "Negative distance improvement is expected due to robots' additional travel; the key benefit is time reduction",
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

        # Add cost if available
        if (
            baseline.get("total_cost") is not None
            and netro.get("estimated_cost") is not None
        ):
            report.append(
                f"| Total Cost | {baseline['total_cost']:.2f} | {netro['estimated_cost']:.2f} | {(baseline['total_cost'] - netro['estimated_cost']) / baseline['total_cost'] * 100:.2f}% |"
            )

        report.append(
            f"| Number of Trucks | {baseline['num_trucks_used']} | {netro['num_trucks_used']} | - |"
        )
        report.append("")

        # Add note about distance calculation
        if improvement["distance_percent"] < 0:
            report.append(
                "**Note about distance:** While the total distance is higher for Netro, this is expected"
            )
            report.append(
                "because robots travel to each customer individually. The critical advantage is the"
            )
            report.append("significant time reduction due to parallel operations.")
            report.append("")

        report.append("## Netro Additional Metrics")
        report.append("")
        report.append(f"- Number of Clusters: {netro['num_clusters']}")
        report.append(f"- Truck Distance: {netro['total_truck_distance']:.2f} km")
        report.append(f"- Robot Distance: {netro['total_robot_distance']:.2f} km")

        # Add explanation of parallel computation
        report.append("")
        report.append("## Time Calculation Method")
        report.append("")
        report.append(
            "Netro time is calculated using a parallel computation model where:"
        )
        report.append("- Trucks travel between cluster centroids")
        report.append("- At each cluster, robots operate in parallel")
        report.append("- The total operation time accounts for this parallelization")
        report.append("")
        report.append(
            "This reflects the real-world advantage of the Netro system, where"
        )
        report.append(
            "multiple deliveries can be made simultaneously by robots at each cluster."
        )

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

        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        # Plot 1: Time and distance comparison
        x = np.arange(len(metrics))
        width = 0.35

        axs[0].bar(
            x - width / 2, traditional_values, width, label="Traditional Truck-Only"
        )
        axs[0].bar(x + width / 2, netro_values, width, label="Netro Hybrid")

        axs[0].set_xticks(x)
        axs[0].set_xticklabels(metrics)
        axs[0].set_ylabel("Value")
        axs[0].set_title("Comparison of Traditional vs Netro Delivery")
        axs[0].legend()

        # Add value labels
        for i, v in enumerate(traditional_values):
            axs[0].text(
                i - width / 2,
                v + 0.05 * max(traditional_values),
                f"{v:.1f}",
                ha="center",
                va="bottom",
            )
        for i, v in enumerate(netro_values):
            axs[0].text(
                i + width / 2,
                v + 0.05 * max(traditional_values),
                f"{v:.1f}",
                ha="center",
                va="bottom",
            )

        # Plot 2: Detailed distance breakdown for Netro
        netro_distances = ["Truck Distance", "Robot Distance", "Total Distance"]
        distances = [
            netro["total_truck_distance"],
            netro["total_robot_distance"],
            netro["total_distance"],
        ]

        bars = axs[1].bar(netro_distances, distances)
        axs[1].set_ylabel("Distance (km)")
        axs[1].set_title("Netro Distance Breakdown")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axs[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 5,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        return fig
