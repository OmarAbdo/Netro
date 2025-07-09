# Netro vs Traditional Truck-Only Delivery Comparison Report

## Key Metrics

| Metric | Traditional | Netro | Improvement |
|--------|------------|-------|-------------|
| Total Time (hours) | 159.76 | 90.17 (Hybrid: 10.74 + Last-Resort: 79.43) | 43.56% |
| Driver Cost (EUR) | 2396.47 | 1609.08 | 32.86% |
| Total Distance (km) | 556.00 | 1906.88 | -242.96% |
| Total Cost | 3488.24 | 1811.26 | 48.08% |
| Number of Trucks | 3 | 4 | - |

## Operational Analysis

### Traditional Approach
- Operation mode: Sequential customer visits (time is sum across all trucks)
- Total operation time: 159.76 hours
- Number of trucks: 3
- Average time per truck: 53.25 hours
- **Total driver cost: 2396.47 EUR**

### Netro Hybrid Approach
- Operation mode: Parallel trucks to centroids, then parallel robots; sequential last-resort trucks if needed
- Parallel hybrid operation time: 10.74 hours
- Last-resort truck time (sequential): 79.43 hours
- Combined Total Time: 90.17 hours
- Truck travel time component (hybrid part): 7.40 hours
- Max cluster operation time (hybrid part): 8.39 hours
- Number of trucks (main hybrid): 4
- Number of last-resort truck routes: 2
- Number of clusters: 4
- **Total driver cost: 1609.08 EUR**

## Time Calculation Explanation

**Traditional System:**
- Total time: 159.8 hours (sum of all truck routes)
- Average per truck: 53.3 hours
- Driver cost: 159.8 hours × €15/h = €2396

**Netro System:**
- Hybrid parallel time: 10.7 hours (max across main truck routes to centroids + their cluster service time)
- Last-resort truck time: 79.4 hours (sum of sequential last-resort truck routes)
- Combined Total Time: 90.2 hours
- Number of trucks (main hybrid): 4
- Total Driver Cost: €1609.08 (sum of all driver hours for hybrid and last-resort)

## Financial Analysis

The Netro system saves **787.39 EUR** in driver costs,
representing a 32.86% reduction compared to the traditional approach.

This cost reduction is achieved through:
1. **Parallel truck operations**: Multiple trucks work simultaneously
2. **Parallel robot deliveries**: Robots deliver to customers simultaneously within clusters
3. **Strategic positioning**: Trucks only travel to cluster centroids
4. **Reduced total operation time**: From parallel operations

**Note about distance:** The total distance is higher for Netro because robots
must travel from cluster centroids to individual customers. However, the time
and cost savings from parallel operations more than compensate for this increase.

## Netro Additional Metrics

- Number of Clusters: 38
- Truck Distance: 855.49 km
- Robot Distance: 1051.39 km
- Distance Ratio (Robot/Truck): 1.2:1