# Netro vs Traditional Truck-Only Delivery Comparison Report

## Key Metrics

| Metric | Traditional | Netro | Improvement |
|--------|------------|-------|-------------|
| Total Time (hours) | 159.76 | 17.00 (Hybrid: 17.00 + Last-Resort: 0.00) | 89.36% |
| Driver Cost (EUR) | 2396.47 | 637.84 | 73.38% |
| Total Distance (km) | 556.00 | 884.86 | -59.15% |
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
- Parallel hybrid operation time: 17.00 hours
- Truck travel time component (hybrid part): 4.13 hours
- Max cluster operation time (hybrid part): 16.20 hours
- Number of trucks (main hybrid): 3
- Number of clusters: 3
- **Total driver cost: 637.84 EUR**

## Time Calculation Explanation

**Traditional System:**
- Total time: 159.8 hours (sum of all truck routes)
- Average per truck: 53.3 hours
- Driver cost: 159.8 hours × €15/h = €2396

**Netro System:**
- Hybrid parallel time: 17.0 hours (max across main truck routes to centroids + their cluster service time)
- Total Time: 17.0 hours
- Number of trucks (main hybrid): 3
- Total Driver Cost: €637.84 (sum of all driver hours for hybrid and last-resort)

## Financial Analysis

The Netro system saves **1758.63 EUR** in driver costs,
representing a 73.38% reduction compared to the traditional approach.

This cost reduction is achieved through:
1. **Parallel truck operations**: Multiple trucks work simultaneously
2. **Parallel robot deliveries**: Robots deliver to customers simultaneously within clusters
3. **Strategic positioning**: Trucks only travel to cluster centroids
4. **Reduced total operation time**: From parallel operations

**Note about distance:** The total distance is higher for Netro because robots
must travel from cluster centroids to individual customers. However, the time
and cost savings from parallel operations more than compensate for this increase.

## Netro Additional Metrics

- Number of Clusters: 5
- Truck Distance: 243.00 km
- Robot Distance: 641.86 km
- Distance Ratio (Robot/Truck): 2.6:1