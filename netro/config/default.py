# netro/config/default.py
"""
Default configuration values for the Netro system.
"""

# Clustering parameters
CLUSTERING = {
    "min_cluster_size": 5,
    "algorithm": "hdbscan",  # Options: 'hdbscan', 'kmeans'
}

# Vehicle parameters
VEHICLES = {
    "truck": {
        "capacity": 2000.0,
        "speed": 60.0,  # km/h
        "cost_per_distance": 0.5,
        "cost_per_time": 20.0,
        "emissions_per_distance": 120.0,
        "loading_time": 5.0,  # minutes
    },
    "robots": {
        "cartken": {
            "robots_per_truck": 132,  # Number of robots per truck
            "capacity": 300.0,
            "speed": 7.2,
            "cost_per_distance": 0.1,
            "cost_per_time": 5.0,
            "emissions_per_distance": 0.0,
            "battery_capacity": 960.0,
            "recharging_rate": 15.0,
        },
        "starship": {
            "robots_per_truck": 336,  # Number of robots per truck
            "capacity": 10.0,
            "speed": 6.0,
            "cost_per_distance": 0.1,
            "cost_per_time": 5.0,
            "emissions_per_distance": 0.0,
            "battery_capacity": 1080.0,
            "recharging_rate": 2.0,
        },
        "ottonomy": {
            "robots_per_truck": 68,  # Number of robots per truck
            "capacity": 80.0,
            "speed": 6.0,
            "cost_per_distance": 0.1,
            "cost_per_time": 5.0,
            "emissions_per_distance": 0.0,
            "battery_capacity": 720.0,
            "recharging_rate": 3.0,
        },
        "nuro": {
            "robots_per_truck": 11,  # Number of robots per truck
            "capacity": 226.8,
            "speed": 72.0,
            "cost_per_distance": 0.1,
            "cost_per_time": 5.0,
            "emissions_per_distance": 0.0,
            "battery_capacity": 960.0,
            "recharging_rate": 15.0,
        },
    }
}

# Routing parameters
ROUTING = {
    "cvrp": {
        "first_solution_strategy": "PATH_CHEAPEST_ARC",
        "local_search_metaheuristic": "GUIDED_LOCAL_SEARCH",
        "time_limit_seconds": 30,
    },
    "local_search": {
        "max_iterations": 1000,
        "improvement_threshold": 1e-4,
        "max_iterations_without_improvement": 100,
    },
}

# Robot deployment parameters
DEPLOYMENT = {
    "robot_launch_time": 2.0,  # minutes
    "robot_recovery_time": 3.0,  # minutes
    "recharge_time_factor": 1.5,
    "loading_time_per_robot": 1.0,  # minutes
}

# I/O parameters
IO = {
    "dataset_path": "netro/dataset/",
    "output_path": "netro/output/",
}
