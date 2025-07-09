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
    # Starship robot parameters
    "robot": {
        "capacity": 10.0,  # kg
        "speed": 6.0,  # km/h
        "cost_per_distance": 0.1,  # (estimate) per km
        "cost_per_time": 5.0,  # (estimate) per hour
        "emissions_per_distance": 0.0,  # g/km
        "battery_capacity": 1080.0,  # minutes (18 hours)
        "recharging_rate": 2.0,  # minutes per charge unit
    },
    # Ottonomy robot parameters
    # "robot": {
    #     "capacity": 80.0,  # kg
    #     "speed": 6.0,  # km/h
    #     "cost_per_distance": 0.1,  # (estimate) per km
    #     "cost_per_time": 5.0,  # (estimate) per hour
    #     "emissions_per_distance": 0.0,  # electric, zero local emissions
    #     "battery_capacity": 720.0,  # minutes (12 hours)
    #     "recharging_rate": 3.0,  # minutes per charge unit
    # },
    # Cartken robot parameters
    "robot": {
        "capacity": 300.0,  # kg (Hauler model, flat surface)
        "speed": 7.2,  # km/h
        "cost_per_distance": 0.1,  # (estimate) per km
        "cost_per_time": 5.0,  # (estimate) per hour
        "emissions_per_distance": 0.0,
        "battery_capacity": 960.0,  # minutes (16 hours)
        "recharging_rate": 15.0,  # minutes per 1 hour charge (4 hours for full charge)
    },
    # Nuro robot parameters
    "robot": {
        "capacity": 226.8,  # kg (500 lb)
        "speed": 72.0,  # km/h (45 mph, road-legal)
        "cost_per_distance": 0.1,  # (estimate) per km
        "cost_per_time": 5.0,  # (estimate) per hour
        "emissions_per_distance": 0.0,
        "battery_capacity": 960.0,  # minutes (16 hours)
        "recharging_rate": 15.0,  # minutes per 1 hour charge (4 hours for full charge)
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
