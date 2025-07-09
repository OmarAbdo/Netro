# Netro vs Traditional Truck-Only Delivery Comparison Report

## Key Metrics

| Metric | Traditional | Netro | Improvement |
|--------|------------|-------|-------------|
| Total Time (hours) | 159.76 | 29.67 (Hybrid: 29.67 + Last-Resort: 0.00) | 81.43% |
| Driver Cost (EUR) | 2396.47 | 781.89 | 67.37% |
| Total Distance (km) | 556.00 | 1455.37 | -161.76% |
| Total Cost | 3488.24 | 0.00 | 100.00% |
| Number of Trucks | 3 | 3 | - |

## Operational Analysis

### Traditional Approach
- Operation mode: Sequential customer visits (time is sum across all trucks)
- Total operation time: 159.76 hours
- Number of trucks: 3
- Average time per truck: 53.25 hours
- **Total driver cost: 2396.47 EUR**

### Netro Hybrid Approach
- Operation mode: Parallel trucks to centroids, then parallel robots; sequential last-resort trucks if needed
- Parallel hybrid operation time: 29.67 hours
- Truck travel time component (hybrid part): 4.38 hours
- Max cluster operation time (hybrid part): 28.74 hours
- Number of trucks (main hybrid): 3
- Number of clusters: 3
- **Total driver cost: 781.89 EUR**

## Time Calculation Explanation

**Traditional System:**
- Total time: 159.8 hours (sum of all truck routes)
- Average per truck: 53.3 hours
- Driver cost: 159.8 hours × €15/h = €2396

**Netro System:**
- Hybrid parallel time: 29.7 hours (max across main truck routes to centroids + their cluster service time)
- Total Time: 29.7 hours
- Number of trucks (main hybrid): 3
- Total Driver Cost: €781.89 (sum of all driver hours for hybrid and last-resort)

## Financial Analysis

The Netro system saves **1614.58 EUR** in driver costs,
representing a 67.37% reduction compared to the traditional approach.

This cost reduction is achieved through:
1. **Parallel truck operations**: Multiple trucks work simultaneously
2. **Parallel robot deliveries**: Robots deliver to customers simultaneously within clusters
3. **Strategic positioning**: Trucks only travel to cluster centroids
4. **Reduced total operation time**: From parallel operations

**Note about distance:** The total distance is higher for Netro because robots
must travel from cluster centroids to individual customers. However, the time
and cost savings from parallel operations more than compensate for this increase.

## Netro Additional Metrics

- Number of Clusters: 7
- Truck Distance: 257.00 km
- Robot Distance: 1198.37 km
- Distance Ratio (Robot/Truck): 4.7:1