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
        Compare baseline and Netro solutions with corrected logic.

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

        # Use the parallel time calculation for Netro, including last-resort if applicable
        netro_parallel_time = netro_solution["parallel_time"] # Time for the hybrid part
        netro_last_resort_time = netro_solution.get("last_resort_truck_time", 0.0)
        # The 'total_time_with_last_resort' key should exist from NetroApplication
        netro_time = netro_solution.get("total_time_with_last_resort", netro_parallel_time + netro_last_resort_time)


        netro_truck_distance = netro_solution["total_truck_distance"] # Already includes last-resort
        netro_robot_distance = netro_solution["total_robot_distance"]
        netro_total_distance = netro_truck_distance + netro_robot_distance
        
        # Number of trucks used in the main hybrid part
        netro_main_truck_routes = netro_solution.get("truck_routes", [])
        netro_num_main_trucks = len(netro_main_truck_routes)
        # Number of trucks used in last-resort (can be from the same pool or additional)
        # For simplicity, we'll report the number of main trucks, as last-resort reuses them.
        # A more detailed metric could be 'total_truck_deployments'.
        netro_num_trucks_reported = netro_num_main_trucks


        # CORRECTED: Calculate driver costs properly
        baseline_driver_cost = self._calculate_baseline_driver_cost(baseline_solution)
        netro_driver_cost = self._calculate_netro_driver_cost(netro_solution) # This will need to account for last_resort_truck_time

        # Calculate cost and emissions improvement
        driver_cost_improvement = (
            (baseline_driver_cost - netro_driver_cost) / baseline_driver_cost * 100
            if baseline_driver_cost > 0
            else 0
        )

        # Use actual cost and emissions from netro_solution if available (they now include last-resort)
        netro_cost = netro_solution.get("total_cost", 0) 
        netro_emissions = netro_solution.get("total_emissions", 0)


        # Calculate sequential time equivalent (sum of all cluster times)
        netro_sequential_time = sum(
            metrics.get("max_time", 0) 
            for metrics in netro_solution["cluster_metrics"].values()
        )
        
        # Calculate improvement percentages using the appropriate Netro time
        time_improvement = (
            (baseline_time - netro_time) / baseline_time * 100 # netro_time now includes last_resort
            if baseline_time > 0
            else 0
        )
        distance_improvement = (
            (baseline_distance - netro_total_distance) / baseline_distance * 100
            if baseline_distance > 0
            else 0
        )
        time_savings = netro_sequential_time - netro_parallel_time
        time_savings_percent = (time_savings / netro_sequential_time * 100) if netro_sequential_time > 0 else 0

        # Create comparison dictionary
        comparison = {
            "baseline": {
                "total_time": baseline_time,
                "total_distance": baseline_distance,
                "total_emissions": baseline_emissions,
                "num_trucks_used": baseline_num_trucks,
                "driver_cost": baseline_driver_cost,
            },
            "netro": {
                "total_time": netro_time, # This is total_time_with_last_resort
                "parallel_time_hybrid": netro_parallel_time, # Time of just the hybrid part
                "sequential_time_equivalent": netro_sequential_time,
                "last_resort_truck_time": netro_last_resort_time,
                "total_truck_distance": netro_truck_distance,
                "total_robot_distance": netro_robot_distance,
                "total_distance": netro_total_distance,
                "total_emissions": netro_emissions, # Changed from estimated_emissions
                "num_trucks_used": netro_num_trucks_reported,
                "num_clusters": len(netro_solution.get("cluster_routes", {})),
                "parallel_computation": True,
                "driver_cost": netro_driver_cost,
                "last_resort_routes_exist": bool(netro_solution.get("last_resort_truck_routes")),
            },
            "improvement": {
                "time_percent": time_improvement,
                "distance_percent": distance_improvement,
                "driver_cost_percent": driver_cost_improvement,
                "driver_cost_absolute": baseline_driver_cost - netro_driver_cost,
                "time_savings": time_savings,
                "time_savings_percent": time_savings_percent,
                "notes": "Corrected driver cost calculation: baseline uses total time, Netro uses parallel time",
            },
            "detailed_breakdown": {
                "baseline_breakdown": self._get_baseline_cost_breakdown(
                    baseline_solution
                ),
                "netro_breakdown": self._get_netro_cost_breakdown(netro_solution),
            },
        }

        return comparison

    def _calculate_baseline_driver_cost(
        self, baseline_solution: Dict[str, Any]
    ) -> float:
        """
        Calculate baseline driver costs CORRECTLY.

        Traditional system: total_time is the sum of all truck operation times.
        Driver cost = total_time * hourly_rate (NOT * number_of_trucks)
        """
        # CORRECTED: The total_time already represents the sum across all trucks
        total_time = baseline_solution["total_time"]  # e.g., 363.91 hours
        return total_time * self.driver_hourly_cost  # 363.91 * 15 = €5,458.65

    def _calculate_netro_driver_cost(self, netro_solution: Dict[str, Any]) -> float:
        """
        Calculate Netro driver costs correctly.

        CORRECTED LOGIC: Even though trucks operate in parallel, we must count
        ALL driver hours worked, not just the maximum time.

        If 2 trucks work 1 hour each in parallel = 2 driver-hours of labor cost.
        """
        # Get individual truck route times from the debug output
        truck_metrics = netro_solution.get("truck_metrics", {})
        cluster_metrics = netro_solution.get("cluster_metrics", {})

        total_driver_hours = 0.0

        # Calculate actual driver hours: sum of (truck_travel + robot_waiting) for each truck in the hybrid part
        for truck_idx, metrics in cluster_metrics.items():
            # Each truck's total time = truck travel + robot waiting
            # This logic for individual truck time in hybrid part might need refinement
            # if netro_solution["truck_metrics"]["route_times_individual"] is available.
            # For now, using the existing _get_truck_route_time which averages.
            truck_route_time_hybrid = self._get_truck_route_time(truck_idx, netro_solution, is_hybrid_part=True)
            cluster_operation_time = (
                metrics.get("max_time", 0.0) # Assuming this is already in hours from NetroRoutingService
            ) 

            total_truck_time_hybrid = truck_route_time_hybrid + cluster_operation_time
            total_driver_hours += total_truck_time_hybrid
        
        # Add driver hours for last-resort truck routes
        # last_resort_truck_time is the sum of durations of all last-resort routes
        total_driver_hours += netro_solution.get("last_resort_truck_time", 0.0)

        return total_driver_hours * self.driver_hourly_cost

    def _get_truck_route_time(
        self, truck_idx: int, netro_solution: Dict[str, Any], is_hybrid_part: bool = False
    ) -> float:
        """Extract truck travel time for a specific truck route."""
        # This is a simplified calculation - in a real system we'd track individual route times
        # If is_hybrid_part, use truck_metrics from the main hybrid solution
        # Otherwise, this function might not be directly applicable for last-resort if times are already summed.
        
        if is_hybrid_part:
            truck_metrics = netro_solution.get("truck_metrics", {})
            # Check if individual route times are available (preferred)
            # This depends on what NetroRoutingService populates in truck_metrics
            # Assuming 'route_times_individual' might be a list of times for each truck route to centroids
            individual_route_times = truck_metrics.get("route_times_individual")
            if individual_route_times and truck_idx < len(individual_route_times):
                return individual_route_times[truck_idx]
            
            # Fallback to average if individual times not available
            total_truck_travel_time_hybrid = truck_metrics.get("total_travel_time", 0.0) # Assuming this is just travel
            num_hybrid_routes = len(netro_solution.get("truck_routes", []))
            return (total_truck_travel_time_hybrid / num_hybrid_routes) if num_hybrid_routes > 0 else 0.0
        
        # For non-hybrid (e.g. last-resort), this function might not be the right place,
        # as last_resort_truck_time is already a sum.
        return 0.0


    def _get_baseline_cost_breakdown(
        self, truck_idx: int, netro_solution: Dict[str, Any]
    ) -> float:
        """Extract truck travel time for a specific truck route."""
        # This is a simplified calculation - in a real system we'd track individual route times
        truck_metrics = netro_solution.get("truck_metrics", {})
        total_truck_time = truck_metrics.get("total_time", 0.0)
        num_routes = len(netro_solution.get("truck_routes", []))

        # Estimate individual truck time as proportional share
        return (total_truck_time / num_routes) if num_routes > 0 else 0.0

    def _get_baseline_cost_breakdown(
        self, baseline_solution: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get detailed cost breakdown for baseline solution."""
        total_time = baseline_solution["total_time"]
        num_trucks = baseline_solution["metrics"]["num_trucks_used"]

        return {
            "total_operation_time": total_time,
            "number_of_trucks": num_trucks,
            "driver_cost": total_time * self.driver_hourly_cost,  # CORRECTED
            "average_time_per_truck": total_time / num_trucks if num_trucks > 0 else 0,
            "operation_mode": "Sequential customer visits (time is sum across all trucks)",
        }

    def _get_netro_cost_breakdown(
        self, netro_solution: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get detailed cost breakdown for Netro solution."""
        truck_metrics = netro_solution.get("truck_metrics", {})
        cluster_metrics = netro_solution.get("cluster_metrics", {})
        parallel_time = netro_solution["parallel_time"]
        num_trucks = len(netro_solution["truck_routes"])

        # Extract truck travel time
        truck_travel_time = truck_metrics.get("total_time", 0)

        # Extract robot operation times
        cluster_operation_times = [
            m.get("max_time", 0) for m in cluster_metrics.values()
        ]
        max_cluster_time = (
            max(cluster_operation_times) if cluster_operation_times else 0
        )

        # Number of clusters
        num_clusters = len(
            [m for m in cluster_metrics.values() if m.get("max_time", 0) > 0]
        )

        return {
            "parallel_operation_time": parallel_time,
            "truck_travel_time": truck_travel_time,
            "max_cluster_operation_time": max_cluster_time, # Max time spent by any truck at a cluster
            "number_of_trucks": num_trucks, # Number of trucks used in hybrid part
            "number_of_clusters": num_clusters,
            "driver_cost_hybrid_part": (netro_solution["parallel_time"] * num_trucks * self.driver_hourly_cost) if num_trucks > 0 else 0, # Cost for the parallel hybrid part
            "driver_cost_last_resort": netro_solution.get("last_resort_truck_time", 0.0) * self.driver_hourly_cost,
            "total_driver_cost": netro_solution.get("driver_cost",0.0), # This should be calculated by _calculate_netro_driver_cost
            "operation_mode": "Parallel trucks to centroids, then parallel robots; sequential last-resort trucks if needed",
            "cost_per_truck_hybrid_avg": (netro_solution["parallel_time"] * self.driver_hourly_cost) if num_trucks > 0 else 0,
            "last_resort_truck_time_sum": netro_solution.get("last_resort_truck_time", 0.0),
            "num_last_resort_routes": len(netro_solution.get("last_resort_truck_routes", [])),
        }

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

        # Get detailed breakdowns if available
        baseline_breakdown = comparison.get("detailed_breakdown", {}).get(
            "baseline_breakdown", {}
        )
        netro_breakdown = comparison.get("detailed_breakdown", {}).get(
            "netro_breakdown", {}
        )

        report = []
        report.append("# Netro vs Traditional Truck-Only Delivery Comparison Report")
        report.append("")
        report.append("## Key Metrics")
        report.append("")
        report.append("| Metric | Traditional | Netro | Improvement |")
        report.append("|--------|------------|-------|-------------|")
        report.append(
            f"| Total Time (hours) | {baseline['total_time']:.2f} | {netro['total_time']:.2f} (Hybrid: {netro.get('parallel_time_hybrid', 0.0):.2f} + Last-Resort: {netro.get('last_resort_truck_time', 0.0):.2f}) | {improvement['time_percent']:.2f}% |"
        )
        report.append(
            f"| Sequential Time Equivalent | - | {netro['sequential_time_equivalent']:.2f} | - |"
        )
        report.append(
            f"| Time Savings from Parallelization | - | {improvement['time_savings']:.2f}h ({improvement['time_savings_percent']:.1f}%) | - |"
        )
        report.append(
            f"| Driver Cost (EUR) | {baseline['driver_cost']:.2f} | {netro['driver_cost']:.2f} | {improvement['driver_cost_percent']:.2f}% |"
        )
        report.append(
            f"| Total Distance (km) | {baseline['total_distance']:.2f} | {netro['total_distance']:.2f} | {improvement['distance_percent']:.2f}% |"
        )

        # Total cost field removed per user request

        report.append(
            f"| Number of Trucks | {baseline['num_trucks_used']} | {netro['num_trucks_used']} | - |" # netro_num_trucks_used is main trucks
        )
        report.append("")

        # Add detailed operational analysis
        if netro_breakdown and baseline_breakdown:
            report.append("## Operational Analysis")
            report.append("")
            report.append("### Traditional Approach")
            report.append(
                f"- Operation mode: {baseline_breakdown.get('operation_mode', 'Sequential')}"
            )
            report.append(
                f"- Total operation time: {baseline_breakdown.get('total_operation_time', 0):.2f} hours"
            )
            report.append(
                f"- Number of trucks: {baseline_breakdown.get('number_of_trucks', 0)}"
            )
            report.append(
                f"- Average time per truck: {baseline_breakdown.get('average_time_per_truck', 0):.2f} hours"
            )
            report.append(f"- **Total driver cost: {baseline['driver_cost']:.2f} EUR**")
            report.append("")

            report.append("### Netro Hybrid Approach")
            report.append(
                f"- Operation mode: {netro_breakdown.get('operation_mode', 'Parallel with potential sequential last-resort')}"
            )
            report.append(
                f"- Parallel hybrid operation time: {netro.get('parallel_time_hybrid', 0.0):.2f} hours"
            )
            if netro.get('last_resort_routes_exist', False):
                report.append(
                    f"- Last-resort truck time (sequential): {netro.get('last_resort_truck_time', 0.0):.2f} hours"
                )
                report.append(
                    f"- Combined Total Time: {netro['total_time']:.2f} hours"
                )
            report.append(
                f"- Truck travel time component (hybrid part): {netro_breakdown.get('truck_travel_time', 0):.2f} hours"
            )
            report.append(
                f"- Max cluster operation time (hybrid part): {netro_breakdown.get('max_cluster_operation_time', 0):.2f} hours"
            )
            report.append(
                f"- Number of trucks (main hybrid): {netro_breakdown.get('number_of_trucks', 0)}"
            )
            if netro_breakdown.get('num_last_resort_routes', 0) > 0:
                 report.append(
                    f"- Number of last-resort truck routes: {netro_breakdown.get('num_last_resort_routes', 0)}"
                )
            report.append(
                f"- Number of clusters: {netro_breakdown.get('number_of_clusters', 0)}"
            )
            # report.append(
            #     f"- Cost per truck (hybrid_avg): {netro_breakdown.get('cost_per_truck_hybrid_avg', 0):.2f} EUR"
            # )
            report.append(f"- **Total driver cost: {netro['driver_cost']:.2f} EUR**")
            report.append("")

        # Add clarification of time calculations
        report.append("## Time Calculation Explanation")
        report.append("")
        report.append("**Traditional System:**")
        report.append(
            f"- Total time: {baseline['total_time']:.1f} hours (sum of all truck routes)"
        )
        report.append(
            f"- Average per truck: {baseline['total_time']/baseline['num_trucks_used']:.1f} hours"
        )
        report.append(
            f"- Driver cost: {baseline['total_time']:.1f} hours × €15/h = €{baseline['driver_cost']:.0f}"
        )
        report.append("")
        report.append("**Netro System:**")
        report.append(
            f"- Hybrid parallel time: {netro.get('parallel_time_hybrid', 0.0):.1f} hours (max across main truck routes to centroids + their cluster service time)"
        )
        if netro.get('last_resort_routes_exist', False):
            report.append(
                f"- Last-resort truck time: {netro.get('last_resort_truck_time', 0.0):.1f} hours (sum of sequential last-resort truck routes)"
            )
            report.append(
                f"- Combined Total Time: {netro['total_time']:.1f} hours"
            )
        else:
            report.append(
                f"- Total Time: {netro['total_time']:.1f} hours"
            )
        report.append(
            f"- Number of trucks (main hybrid): {netro['num_trucks_used']}"
        )
        # Driver cost calculation is now more complex due to summing individual efforts + last resort
        report.append(
            f"- Total Driver Cost: €{netro['driver_cost']:.2f} (sum of all driver hours for hybrid and last-resort)"
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
        report.append("This cost reduction is achieved through:")
        report.append(
            "1. **Parallel truck operations**: Multiple trucks work simultaneously"
        )
        report.append(
            "2. **Parallel robot deliveries**: Robots deliver to customers simultaneously within clusters"
        )
        report.append(
            "3. **Strategic positioning**: Trucks only travel to cluster centroids"
        )
        report.append("4. **Reduced total operation time**: From parallel operations")
        report.append("")

        # Add note about distance calculation
        if improvement["distance_percent"] < 0:
            report.append(
                "**Note about distance:** The total distance is higher for Netro because robots"
            )
            report.append(
                "must travel from cluster centroids to individual customers. However, the time"
            )
            report.append(
                "and cost savings from parallel operations more than compensate for this increase."
            )
            report.append("")

        report.append("## Netro Additional Metrics")
        report.append("")
        report.append(f"- Number of Clusters: {netro['num_clusters']}")
        report.append(f"- Truck Distance: {netro['total_truck_distance']:.2f} km")
        report.append(f"- Robot Distance: {netro['total_robot_distance']:.2f} km")
        report.append(
            f"- Distance Ratio (Robot/Truck): {netro['total_robot_distance']/netro['total_truck_distance']:.1f}:1"
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
        # Use the combined time for Netro, which includes last-resort if applicable
        netro_display_time = netro.get("total_time", netro.get("parallel_time_hybrid", 0.0))
        values1 = [[baseline["total_time"]], [netro_display_time]]

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
        axs[0, 1].set_title("Driver Cost Comparison (Fixed)")
        axs[0, 1].set_xticks(x2)
        axs[0, 1].set_xticklabels(metrics2)
        axs[0, 1].legend()

        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            axs[0, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.02,
                f"{height:.0f}",
                ha="center",
                va="bottom",
            )
        for bar in bars4:
            height = bar.get_height()
            axs[0, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.02,
                f"{height:.0f}",
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
                height + 50,
                f"{height:.0f}",
                ha="center",
                va="bottom",
            )
        for bar in bars6:
            height = bar.get_height()
            axs[1, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 50,
                f"{height:.0f}",
                ha="center",
                va="bottom",
            )

        # Plot 4: Time breakdown explanation (bottom-right)
        baseline_breakdown = comparison.get("detailed_breakdown", {}).get(
            "baseline_breakdown", {}
        )
        netro_breakdown = comparison.get("detailed_breakdown", {}).get(
            "netro_breakdown", {}
        )

        # Show time components for Netro if last resort routes exist
        if netro.get("last_resort_routes_exist", False):
            time_categories = ["Traditional\n(Sum of routes)", "Netro Hybrid\n(Parallel Part)", "Netro Last-Resort\n(Sequential)", "Netro Combined"]
            time_values = [
                baseline["total_time"], 
                netro.get("parallel_time_hybrid", 0.0), 
                netro.get("last_resort_truck_time", 0.0),
                netro.get("total_time", 0.0) # Combined time
            ]
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
            axs[1, 1].set_title("Time Components")
        else:
            time_categories = ["Traditional\n(Sum of routes)", "Netro\n(Parallel Time)"]
            time_values = [baseline["total_time"], netro.get("total_time", 0.0)] # Should be parallel_time_hybrid if no last_resort
            colors = ["#1f77b4", "#ff7f0e"]
            axs[1, 1].set_title("Time Calculation: Sum vs Parallel")
        
        bars7 = axs[1, 1].bar(time_categories, time_values, color=colors)
        axs[1, 1].set_ylabel("Hours")
        
        # Add value labels
        for bar in bars7:
            height = bar.get_height()
            axs[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 5,
                f"{height:.1f}h",
                ha="center",
                va="bottom",
            )

        # Add improvement percentages as text
        fig.text(
            0.5,
            0.02,
            f"Time improvement: {comparison['improvement']['time_percent']:.1f}% | "
            f"Driver cost savings: {comparison['improvement']['driver_cost_percent']:.1f}% | "
            f"Distance change: {comparison['improvement']['distance_percent']:.1f}%",
            ha="center",
            fontsize=12,
            weight="bold",
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        return fig
