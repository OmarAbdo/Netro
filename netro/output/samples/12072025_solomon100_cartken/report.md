# Netro vs Traditional Truck-Only Delivery Comparison Report

## Key Metrics

| Metric | Traditional | Netro | Improvement |
|--------|------------|-------|-------------|
| Total Time (hours) | 164.14 | 10.33 (Hybrid: 10.33 + Last-Resort: 0.00) | 93.71% |
| Sequential Time Equivalent | - | 82.73 | - |
| Time Savings from Parallelization | - | 72.41h (87.5%) | - |
| Driver Cost (EUR) | 2462.16 | 1240.99 | 49.60% |
| Total Distance (km) | 818.00 | 1072.09 | -31.06% |
| Number of Trucks | 10 | 11 | - |

## Operational Analysis

### Traditional Approach
- Operation mode: Sequential customer visits (time is sum across all trucks)
- Total operation time: 164.14 hours
- Number of trucks: 10
- Average time per truck: 16.41 hours
- **Total driver cost: 2462.16 EUR**

### Netro Hybrid Approach
- Operation mode: Parallel trucks to centroids, then parallel robots; sequential last-resort trucks if needed
- Parallel hybrid operation time: 10.33 hours
- Truck travel time component (hybrid part): 11.81 hours
- Max cluster operation time (hybrid part): 8.70 hours
- Number of trucks (main hybrid): 11
- Number of clusters: 10
- **Total driver cost: 1240.99 EUR**

## Time Calculation Explanation

**Traditional System:**
- Total time: 164.1 hours (sum of all truck routes)
- Average per truck: 16.4 hours
- Driver cost: 164.1 hours × €15/h = €2462

**Netro System:**
- Hybrid parallel time: 10.3 hours (max across main truck routes to centroids + their cluster service time)
- Total Time: 10.3 hours
- Number of trucks (main hybrid): 11
- Total Driver Cost: €1240.99 (sum of all driver hours for hybrid and last-resort)

## Financial Analysis

The Netro system saves **1221.17 EUR** in driver costs,
representing a 49.60% reduction compared to the traditional approach.

This cost reduction is achieved through:
1. **Parallel truck operations**: Multiple trucks work simultaneously
2. **Parallel robot deliveries**: Robots deliver to customers simultaneously within clusters
3. **Strategic positioning**: Trucks only travel to cluster centroids
4. **Reduced total operation time**: From parallel operations

**Note about distance:** The total distance is higher for Netro because robots
must travel from cluster centroids to individual customers. However, the time
and cost savings from parallel operations more than compensate for this increase.

## Netro Additional Metrics

- Number of Clusters: 10
- Truck Distance: 699.47 km
- Robot Distance: 372.63 km
- Distance Ratio (Robot/Truck): 0.5:1