# Netro System: Time Calculation Models

This document outlines the mathematical formulas used to calculate the different time metrics within the Netro project. There are three key models: the Baseline (truck-only) model, the Netro Parallel Time (makespan), and the Netro Total Operational Time (for cost analysis).

---

## 1. Baseline Model: Truck-Only Total Time

This model represents the total time taken for a traditional, truck-only delivery system. The total time is the sum of the time for all truck routes, where each route's time is determined by the VRP solver, implicitly including travel and service times.

### Formula

**T_baseline = &Sigma; (T_travel_customer) + &Sigma; (T_service_customer)**

- **T_baseline**: The total time for the entire truck-only delivery operation.
- **&Sigma; (T_travel_customer)**: The sum of travel times between all consecutive customer stops across all truck routes.
- **&Sigma; (T_service_customer)**: The sum of service times at every customer location, as specified in the dataset.

---

## 2. Netro Model 1: Parallel Time (Makespan)

This model calculates the total duration of the hybrid truck-robot operation, from the moment the trucks leave the depot until the last truck completes its final task. Since trucks operate in parallel, the total time is determined by the truck that takes the longest to complete its full tour. This is also known as the makespan of the operation.

### Formula

**T_parallel = max(T_tour_truck_1, T_tour_truck_2, ..., T_tour_truck_N) + T_last_resort**

### Component Breakdown

- **T_parallel**: The total wall-clock time for the Netro hybrid operation.
- **max(...)**: The maximum function, which takes the longest time among all parallel truck tours.
- **T_tour_truck_i**: The total time for a single truck *i*'s complete tour.
  - **T_tour_truck_i = T_truck_travel_i + &Sigma; (T_cluster_op_j)**
    - **T_truck_travel_i**: The time truck *i* spends traveling between the depot and its assigned cluster centroids. `(Distance / Speed)`
    - **&Sigma; (T_cluster_op_j)**: The sum of times the truck *i* is stationary at each cluster *j* it services.
      - **T_cluster_op_j = T_unloading + max(T_robot_1, T_robot_2, ..., T_robot_R)**
        - **T_unloading**: A fixed time from the configuration to unload the robot fleet at a cluster.
        - **max(T_robot_k)**: The time of the longest robot mission launched from that cluster.
          - **T_robot_k = T_robot_travel_k + T_robot_service_k + T_launch + T_recovery + T_recharge_k**
            - `T_robot_travel_k`: Robot travel time `(Distance / Speed)`.
            - `T_robot_service_k`: Service time at customer stops (Note: Currently hardcoded as 5 mins/customer in the code).
            - `T_launch` & `T_recovery`: Fixed times from config for robot deployment and retrieval.
            - `T_recharge_k`: Time spent recharging if the robot's travel time exceeds its battery capacity.
- **T_last_resort**: The time for a separate, sequential truck-only route to service any customers missed by the initial hybrid plan. This time is added on at the end.

---

## 3. Netro Model 2: Total Operational Time (for Costing)

This model calculates the cumulative operational time of all vehicles. It is the sum of the time every truck and every robot was actively working. This metric is essential for accurate cost analysis, as costs are typically tied to the operational hours of each asset.

### Formula

**T_operational_cost = T_all_trucks + T_all_robots**

### Component Breakdown

- **T_operational_cost**: The sum of active operational time for all vehicles in the system.
- **T_all_trucks**: The sum of the travel times of all trucks.
  - **T_all_trucks = &Sigma; (T_truck_travel_i) + T_travel_last_resort**
    - **&Sigma; (T_truck_travel_i)**: The sum of travel times for each individual truck on its primary cluster route.
    - **T_travel_last_resort**: The travel time of the truck(s) used for the final last-resort route.
- **T_all_robots**: The sum of the mission times for every single robot route that was executed.
  - **T_all_robots = &Sigma; (T_robot_k)**
    - **T_robot_k**: The full mission time for an individual robot *k*, calculated exactly as defined in the Parallel Time model above.

---
---

## Appendix: Detailed Formula for a Single Truck's Tour Time (T_tour_truck_i)

This section provides a clear, hierarchical breakdown of the formula for `T_tour_truck_i`, which is a critical component of the main **Parallel Time (Makespan)** model.

### High-Level Formula

The total time for a single truck's tour is the sum of its travel time and the time it spends waiting at each cluster for robot operations to complete.

![T_tour_truck_i = T_truck_travel_i + \sum_{j \in J_i} T_{cluster\_op_j}](https://latex.codecogs.com/svg.image?T_{tour\_truck_i}=T_{truck\_travel_i}+\sum_{j\in&space;J_i}T_{cluster\_op_j})

- **T_tour_truck_i**: The total time for truck *i*'s tour.
- **T_truck_travel_i**: The total time truck *i* spends traveling.
- **J_i**: The set of all clusters *j* assigned to truck *i*.
- **T_cluster_op_j**: The time the truck spends stationary at cluster *j*.

---

### Component Breakdown

#### 1. Truck Travel Time (T_truck_travel_i)

This is the time the truck is physically moving between the depot and the centroids of its assigned clusters.

![T_{truck\_travel_i} = \frac{D_{route_i}}{S_{truck}}](https://latex.codecogs.com/svg.image?T_{truck\_travel_i}=\frac{D_{route_i}}{S_{truck}})

- **D_route_i**: The total distance of truck *i*'s route (Depot -> Cluster 1 -> ... -> Cluster N -> Depot).
- **S_truck**: The speed of the truck.

#### 2. Cluster Operation Time (T_cluster_op_j)

This is the time the truck waits at a single cluster *j*. It's composed of the time to unload the robots plus the time of the longest robot mission launched from that cluster.

![T_{cluster\_op_j} = T_{unloading} + \max(T_{robot_1}, T_{robot_2}, ..., T_{robot_R})](https://latex.codecogs.com/svg.image?T_{cluster\_op_j}=T_{unloading}+\max(T_{robot_1},&space;T_{robot_2},&space;...,&space;T_{robot_R}))

- **T_unloading**: A fixed time from the configuration to unload the robot fleet.
- **max(...)**: The maximum function, as the truck must wait for the last robot to return before it can leave.
- **T_robot_k**: The total mission time for a single robot *k*.

#### 3. Single Robot Mission Time (T_robot_k)

This is the most granular part of the model, representing the total time for one robot's delivery cycle.

![T_{robot_k} = T_{travel_k} + T_{service_k} + T_{launch} + T_{recovery} + T_{recharge_k}](https://latex.codecogs.com/svg.image?T_{robot_k}=T_{travel_k}+T_{service_k}+T_{launch}+T_{recovery}+T_{recharge_k})

- **T_travel_k**: The robot's travel time: `(Distance_k / Speed_robot)`.
- **T_service_k**: The sum of service times at each customer on the robot's route.
- **T_launch**: Fixed time to launch the robot.
- **T_recovery**: Fixed time to recover the robot.
- **T_recharge_k**: Extra time for recharging if the robot's travel time exceeds its battery capacity.
