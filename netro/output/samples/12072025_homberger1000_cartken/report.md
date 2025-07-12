# Netro vs Traditional Truck-Only Delivery Comparison Report

## Key Metrics

| Metric | Traditional | Netro | Improvement |
|--------|------------|-------|-------------|
| Total Time (hours) | 2189.20 | 23.75 (Hybrid: 23.75 + Last-Resort: 0.00) | 98.91% |
| Driver Cost (EUR) | 32837.97 | 11384.02 | 65.33% |
| Total Distance (km) | 40977.00 | 38944.68 | 4.96% |
| Total Cost | 64459.91 | 0.00 | 100.00% |
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
- Parallel hybrid operation time: 23.75 hours
- Truck travel time component (hybrid part): 582.03 hours
- Max cluster operation time (hybrid part): 15.88 hours
- Number of trucks (main hybrid): 85
- Number of clusters: 84
- **Total driver cost: 11384.02 EUR**

## Time Calculation Explanation

**Traditional System:**
- Total time: 2189.2 hours (sum of all truck routes)
- Average per truck: 24.3 hours
- Driver cost: 2189.2 hours × €15/h = €32838

**Netro System:**
- Hybrid parallel time: 23.8 hours (max across main truck routes to centroids + their cluster service time)
- Total Time: 23.8 hours
- Number of trucks (main hybrid): 85
- Total Driver Cost: €11384.02 (sum of all driver hours for hybrid and last-resort)

## Financial Analysis

The Netro system saves **21453.96 EUR** in driver costs,
representing a 65.33% reduction compared to the traditional approach.

This cost reduction is achieved through:
1. **Parallel truck operations**: Multiple trucks work simultaneously
2. **Parallel robot deliveries**: Robots deliver to customers simultaneously within clusters
3. **Strategic positioning**: Trucks only travel to cluster centroids
4. **Reduced total operation time**: From parallel operations

## Netro Additional Metrics

- Number of Clusters: 84
- Truck Distance: 34857.56 km
- Robot Distance: 4087.12 km
- Distance Ratio (Robot/Truck): 0.1:1