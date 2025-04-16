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
        "capacity": 200.0,
        "speed": 60.0,  # km/h
        "cost_per_distance": 0.5,  # per km
        "cost_per_time": 20.0,  # per hour
        "emissions_per_distance": 120.0,  # g/km
        "robot_capacity": 5,  # number of robots
        "loading_time": 5.0,  # minutes
    },
    "robot": {
        "capacity": 30.0,
        "speed": 15.0,  # km/h
        "cost_per_distance": 0.1,  # per km
        "cost_per_time": 5.0,  # per hour
        "emissions_per_distance": 0.0,  # g/km
        "battery_capacity": 120.0,  # minutes
        "recharging_rate": 2.0,  # minutes per charge unit
    },
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
