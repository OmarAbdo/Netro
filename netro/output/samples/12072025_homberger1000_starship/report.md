# Netro vs Traditional Truck-Only Delivery Comparison Report

## Key Metrics

| Metric | Traditional | Netro | Improvement |
|--------|------------|-------|-------------|
| Total Time (hours) | 2189.20 | 1432.73 (Hybrid: 21.37 + Last-Resort: 1411.36) | 34.55% |
| Sequential Time Equivalent | - | 705.35 | - |
| Time Savings from Parallelization | - | 683.98h (97.0%) | - |
| Driver Cost (EUR) | 32837.97 | 31750.71 | 3.31% |
| Total Distance (km) | 40977.00 | 71163.46 | -73.67% |
| Number of Trucks | 90 | 85 | - |

## Operational Analysis

### Traditional Approach
- Operation mode: Sequential customer visits (time is sum across all trucks)
- Total operation time: 2189.20 hours
- Number of trucks: 90
- Average time per truck: 24.32 hours
- **Total driver cost: 32837.97 EUR**

### Netro Hybrid Approach
- Operation mode: Parallel trucks to centroids, then parallel robots; sequential last-resort trucks if needed
- Parallel hybrid operation time: 21.37 hours
- Last-resort truck time (sequential): 1411.36 hours
- Combined Total Time: 1432.73 hours
- Truck travel time component (hybrid part): 582.03 hours
- Max cluster operation time (hybrid part): 15.05 hours
- Number of trucks (main hybrid): 85
- Number of last-resort truck routes: 70
- Number of clusters: 84
- **Total driver cost: 31750.71 EUR**

## Time Calculation Explanation

**Traditional System:**
- Total time: 2189.2 hours (sum of all truck routes)
- Average per truck: 24.3 hours
- Driver cost: 2189.2 hours × €15/h = €32838

**Netro System:**
- Hybrid parallel time: 21.4 hours (max across main truck routes to centroids + their cluster service time)
- Last-resort truck time: 1411.4 hours (sum of sequential last-resort truck routes)
- Combined Total Time: 1432.7 hours
- Number of trucks (main hybrid): 85
- Total Driver Cost: €31750.71 (sum of all driver hours for hybrid and last-resort)

## Financial Analysis

The Netro system saves **1087.26 EUR** in driver costs,
representing a 3.31% reduction compared to the traditional approach.

This cost reduction is achieved through:
1. **Parallel truck operations**: Multiple trucks work simultaneously
2. **Parallel robot deliveries**: Robots deliver to customers simultaneously within clusters
3. **Strategic positioning**: Trucks only travel to cluster centroids
4. **Reduced total operation time**: From parallel operations

**Note about distance:** The total distance is higher for Netro because robots
must travel from cluster centroids to individual customers. However, the time
and cost savings from parallel operations more than compensate for this increase.

## Netro Additional Metrics

- Number of Clusters: 84
- Truck Distance: 66469.56 km
- Robot Distance: 4693.90 km
- Distance Ratio (Robot/Truck): 0.1:1