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

    Based on:
    Ostermeier et al. (2022), "Cost-optimal truck-and-robot routing for last-mile delivery."
    De Maio et al. (2023), "Sustainable last-mile distribution with autonomous robots and public transportation."
    """

    def __init__(self, driver_hourly_cost: float = 15.0):
        """
        Initialize the comparison service.

        Args:
            driver_hourly_cost: Hourly cost for truck drivers in EUR.
        """
        self.driver_hourly_cost = driver_hourly_cost

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
        baseline_num_trucks = baseline_solution["metrics"]["num_trucks_used"]

        # Use the parallel time calculation for Netro, not sequential
        netro_time = (
            netro_solution["parallel_time"]
            if "parallel_time" in netro_solution
            else netro_solution["total_time"]
        )
        netro_truck_distance = netro_solution["total_truck_distance"]
        netro_robot_distance = netro_solution["total_robot_distance"]
        netro_total_distance = netro_truck_distance + netro_robot_distance
        netro_num_trucks = len(netro_solution["truck_routes"])

        # Calculate driver costs - this is the key new metric
        # For baseline: each truck is active for the full time of its route
        baseline_driver_cost = (
            baseline_time * baseline_num_trucks * self.driver_hourly_cost
        )

        # For Netro: trucks only accumulate driver costs for their actual travel time plus waiting time
        # This is a major advantage - trucks spend less time on the road
        netro_driver_cost = 0

        # If truck metrics has individual route times, use them
        if (
            "truck_metrics" in netro_solution
            and "route_times" in netro_solution["truck_metrics"]
        ):
            # Sum up the time for each route
            for route_time in netro_solution["truck_metrics"]["route_times"]:
                netro_driver_cost += route_time * self.driver_hourly_cost
        else:
            # Fallback: estimate based on average speed and distance
            # This is less accurate but still shows the benefit
            netro_driver_cost = (
                netro_solution["truck_metrics"]["total_time"] * self.driver_hourly_cost
            )

        # Add waiting time costs for when trucks are at clusters
        if "cluster_metrics" in netro_solution:
            for truck_idx, metrics in netro_solution["cluster_metrics"].items():
                if "max_time" in metrics:
                    # Driver is waiting while robots deliver
                    netro_driver_cost += metrics["max_time"] * self.driver_hourly_cost

        # Calculate cost and emissions improvement
        driver_cost_improvement = (
            (baseline_driver_cost - netro_driver_cost) / baseline_driver_cost * 100
            if baseline_driver_cost > 0
            else 0
        )

        # Compute other cost and emissions for Netro (simplified)
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

        # Create comparison dictionary
        comparison = {
            "baseline": {
                "total_time": baseline_time,
                "total_distance": baseline_distance,
                "total_cost": baseline_cost,
                "total_emissions": baseline_emissions,
                "num_trucks_used": baseline_num_trucks,
                "driver_cost": baseline_driver_cost,
            },
            "netro": {
                "total_time": netro_time,
                "total_truck_distance": netro_truck_distance,
                "total_robot_distance": netro_robot_distance,
                "total_distance": netro_total_distance,
                "estimated_cost": netro_cost,
                "estimated_emissions": netro_emissions,
                "num_trucks_used": netro_num_trucks,
                "num_clusters": len(netro_solution["cluster_routes"]),
                "parallel_computation": True,  # Flag indicating we're using parallel time
                "driver_cost": netro_driver_cost,
            },
            "improvement": {
                "time_percent": time_improvement,
                "distance_percent": distance_improvement,
                "driver_cost_percent": driver_cost_improvement,
                "driver_cost_absolute": baseline_driver_cost - netro_driver_cost,
                "notes": "Negative distance improvement is expected due to robots' additional travel; the key benefits are time and cost reduction",
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
            f"| Driver Cost (EUR) | {baseline['driver_cost']:.2f} | {netro['driver_cost']:.2f} | {improvement['driver_cost_percent']:.2f}% |"
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

        # Highlight cost savings
        report.append("## Financial Analysis")
        report.append("")
        report.append(
            f"The Netro system saves **{improvement['driver_cost_absolute']:.2f} EUR** in driver costs,"
        )
        report.append(
            f"representing a {improvement['driver_cost_percent']:.2f}% reduction compared to the traditional approach."
        )
        report.append("")
        report.append("This significant cost reduction is achieved through:")
        report.append("1. Reduced active driving time for truck drivers")
        report.append("2. Parallel delivery operations by autonomous robots")
        report.append("3. More efficient routing of trucks between strategic points")
        report.append("")

        # Add note about distance calculation
        if improvement["distance_percent"] < 0:
            report.append(
                "**Note about distance:** While the total distance is higher for Netro, this is expected"
            )
            report.append(
                "because robots travel to each customer individually. The critical advantages are the"
            )
            report.append(
                "significant time reduction and cost savings due to parallel operations."
            )
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
        Generate charts comparing key metrics.

        Args:
            comparison: Comparison data from the compare method.

        Returns:
            Matplotlib figure with comparison plots.
        """
        baseline = comparison["baseline"]
        netro = comparison["netro"]

        # Create a 2x2 grid of subplots for our metrics
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Time comparison (top-left)
        metrics1 = ["Total Time (hours)"]
        values1 = [[baseline["total_time"]], [netro["total_time"]]]

        x1 = np.arange(len(metrics1))
        width = 0.35

        bars1 = axs[0, 0].bar(x1 - width / 2, values1[0], width, label="Traditional")
        bars2 = axs[0, 0].bar(x1 + width / 2, values1[1], width, label="Netro")

        axs[0, 0].set_ylabel("Hours")
        axs[0, 0].set_title("Delivery Time Comparison")
        axs[0, 0].set_xticks(x1)
        axs[0, 0].set_xticklabels(metrics1)
        axs[0, 0].legend()

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            axs[0, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 5,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )
        for bar in bars2:
            height = bar.get_height()
            axs[0, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 5,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )

        # Plot 2: Driver cost comparison (top-right)
        metrics2 = ["Driver Cost (EUR)"]
        values2 = [[baseline["driver_cost"]], [netro["driver_cost"]]]

        x2 = np.arange(len(metrics2))

        bars3 = axs[0, 1].bar(x2 - width / 2, values2[0], width, label="Traditional")
        bars4 = axs[0, 1].bar(x2 + width / 2, values2[1], width, label="Netro")

        axs[0, 1].set_ylabel("EUR")
        axs[0, 1].set_title("Driver Cost Comparison")
        axs[0, 1].set_xticks(x2)
        axs[0, 1].set_xticklabels(metrics2)
        axs[0, 1].legend()

        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            axs[0, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 5,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )
        for bar in bars4:
            height = bar.get_height()
            axs[0, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 5,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )

        # Plot 3: Distance comparison (bottom-left)
        metrics3 = ["Total Distance (km)"]
        values3 = [[baseline["total_distance"]], [netro["total_distance"]]]

        x3 = np.arange(len(metrics3))

        bars5 = axs[1, 0].bar(x3 - width / 2, values3[0], width, label="Traditional")
        bars6 = axs[1, 0].bar(x3 + width / 2, values3[1], width, label="Netro")

        axs[1, 0].set_ylabel("Kilometers")
        axs[1, 0].set_title("Distance Comparison")
        axs[1, 0].set_xticks(x3)
        axs[1, 0].set_xticklabels(metrics3)
        axs[1, 0].legend()

        # Add value labels
        for bar in bars5:
            height = bar.get_height()
            axs[1, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 5,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )
        for bar in bars6:
            height = bar.get_height()
            axs[1, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 5,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )

        # Plot 4: Netro distance breakdown (bottom-right)
        metrics4 = ["Truck Distance", "Robot Distance", "Total Distance"]
        values4 = [
            netro["total_truck_distance"],
            netro["total_robot_distance"],
            netro["total_distance"],
        ]

        bars7 = axs[1, 1].bar(metrics4, values4)

        axs[1, 1].set_ylabel("Kilometers")
        axs[1, 1].set_title("Netro Distance Breakdown")

        # Add value labels
        for bar in bars7:
            height = bar.get_height()
            axs[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 5,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )

        # Add improvement percentages as text
        fig.text(
            0.5,
            0.02,
            f"Time improvement: {comparison['improvement']['time_percent']:.2f}% | "
            f"Cost savings: {comparison['improvement']['driver_cost_percent']:.2f}% | "
            f"Distance change: {comparison['improvement']['distance_percent']:.2f}%",
            ha="center",
            fontsize=12,
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        return fig
