# Netro vs Traditional Truck-Only Delivery Comparison Report

## Key Metrics

| Metric | Traditional | Netro | Improvement |
|--------|------------|-------|-------------|
| Total Time (hours) | 2189.27 | 13.20 (Hybrid: 13.20 + Last-Resort: 0.00) | 99.40% |
| Sequential Time Equivalent | - | 141.62 | - |
| Time Savings from Parallelization | - | 128.42h (90.7%) | - |
| Driver Cost (EUR) | 32839.03 | 2124.26 | 93.53% |
| Total Distance (km) | 40980.00 | 38986.95 | 4.86% |
| Number of Trucks | 90 | 85 | - |

## Operational Analysis

### Traditional Approach
- Operation mode: Sequential customer visits (time is sum across all trucks)
- Total operation time: 2189.27 hours
- Number of trucks: 90
- Average time per truck: 24.33 hours
- **Total driver cost: 32839.03 EUR**

### Netro Hybrid Approach
- Operation mode: Parallel trucks to centroids, then parallel robots; sequential last-resort trucks if needed
- Parallel hybrid operation time: 13.20 hours
- Truck travel time component (hybrid part): 582.03 hours
- Max cluster operation time (hybrid part): 2.37 hours
- Number of trucks (main hybrid): 85
- Number of clusters: 84
- **Total driver cost: 2124.26 EUR**

## Time Calculation Explanation

**Traditional System:**
- Total time: 2189.3 hours (sum of all truck routes)
- Average per truck: 24.3 hours
- Driver cost: 2189.3 hours × €15/h = €32839

**Netro System:**
- Hybrid parallel time: 13.2 hours (max across main truck routes to centroids + their cluster service time)
- Total Time: 13.2 hours
- Number of trucks (main hybrid): 85
- Total Driver Cost: €2124.26 (sum of all driver hours for hybrid and last-resort)

## Financial Analysis

The Netro system saves **30714.77 EUR** in driver costs,
representing a 93.53% reduction compared to the traditional approach.

This cost reduction is achieved through:
1. **Parallel truck operations**: Multiple trucks work simultaneously
2. **Parallel robot deliveries**: Robots deliver to customers simultaneously within clusters
3. **Strategic positioning**: Trucks only travel to cluster centroids
4. **Reduced total operation time**: From parallel operations

## Netro Additional Metrics

- Number of Clusters: 84
- Truck Distance: 34857.56 km
- Robot Distance: 4129.38 km
- Distance Ratio (Robot/Truck): 0.1:1