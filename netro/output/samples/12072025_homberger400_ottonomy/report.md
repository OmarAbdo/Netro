# Netro vs Traditional Truck-Only Delivery Comparison Report

## Key Metrics

| Metric | Traditional | Netro | Improvement |
|--------|------------|-------|-------------|
| Total Time (hours) | 721.07 | 13.76 (Hybrid: 13.76 + Last-Resort: 0.00) | 98.09% |
| Sequential Time Equivalent | - | 197.73 | - |
| Time Savings from Parallelization | - | 183.97h (93.0%) | - |
| Driver Cost (EUR) | 10816.04 | 2966.00 | 72.58% |
| Total Distance (km) | 7119.00 | 7154.99 | -0.51% |
| Number of Trucks | 36 | 34 | - |

## Operational Analysis

### Traditional Approach
- Operation mode: Sequential customer visits (time is sum across all trucks)
- Total operation time: 721.07 hours
- Number of trucks: 36
- Average time per truck: 20.03 hours
- **Total driver cost: 10816.04 EUR**

### Netro Hybrid Approach
- Operation mode: Parallel trucks to centroids, then parallel robots; sequential last-resort trucks if needed
- Parallel hybrid operation time: 13.76 hours
- Truck travel time component (hybrid part): 86.09 hours
- Max cluster operation time (hybrid part): 10.64 hours
- Number of trucks (main hybrid): 34
- Number of clusters: 33
- **Total driver cost: 2966.00 EUR**

## Time Calculation Explanation

**Traditional System:**
- Total time: 721.1 hours (sum of all truck routes)
- Average per truck: 20.0 hours
- Driver cost: 721.1 hours × €15/h = €10816

**Netro System:**
- Hybrid parallel time: 13.8 hours (max across main truck routes to centroids + their cluster service time)
- Total Time: 13.8 hours
- Number of trucks (main hybrid): 34
- Total Driver Cost: €2966.00 (sum of all driver hours for hybrid and last-resort)

## Financial Analysis

The Netro system saves **7850.05 EUR** in driver costs,
representing a 72.58% reduction compared to the traditional approach.

This cost reduction is achieved through:
1. **Parallel truck operations**: Multiple trucks work simultaneously
2. **Parallel robot deliveries**: Robots deliver to customers simultaneously within clusters
3. **Strategic positioning**: Trucks only travel to cluster centroids
4. **Reduced total operation time**: From parallel operations

**Note about distance:** The total distance is higher for Netro because robots
must travel from cluster centroids to individual customers. However, the time
and cost savings from parallel operations more than compensate for this increase.

## Netro Additional Metrics

- Number of Clusters: 33
- Truck Distance: 5143.30 km
- Robot Distance: 2011.69 km
- Distance Ratio (Robot/Truck): 0.4:1