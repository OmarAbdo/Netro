# Netro vs Traditional Truck-Only Delivery Comparison Report

## Key Metrics

| Metric | Traditional | Netro | Improvement |
|--------|------------|-------|-------------|
| Total Time (hours) | 2190.79 | 1432.67 (Hybrid: 21.37 + Last-Resort: 1411.29) | 34.61% |
| Driver Cost (EUR) | 32861.87 | 31749.67 | 3.38% |
| Total Distance (km) | 41071.00 | 71159.46 | -73.26% |
| Total Cost | 64539.58 | 44149.63 | 31.59% |
| Number of Trucks | 90 | 85 | - |

## Operational Analysis

### Traditional Approach
- Operation mode: Sequential customer visits (time is sum across all trucks)
- Total operation time: 2190.79 hours
- Number of trucks: 90
- Average time per truck: 24.34 hours
- **Total driver cost: 32861.87 EUR**

### Netro Hybrid Approach
- Operation mode: Parallel trucks to centroids, then parallel robots; sequential last-resort trucks if needed
- Parallel hybrid operation time: 21.37 hours
- Last-resort truck time (sequential): 1411.29 hours
- Combined Total Time: 1432.67 hours
- Truck travel time component (hybrid part): 582.03 hours
- Max cluster operation time (hybrid part): 15.05 hours
- Number of trucks (main hybrid): 85
- Number of last-resort truck routes: 70
- Number of clusters: 84
- **Total driver cost: 31749.67 EUR**

## Time Calculation Explanation

**Traditional System:**
- Total time: 2190.8 hours (sum of all truck routes)
- Average per truck: 24.3 hours
- Driver cost: 2190.8 hours × €15/h = €32862

**Netro System:**
- Hybrid parallel time: 21.4 hours (max across main truck routes to centroids + their cluster service time)
- Last-resort truck time: 1411.3 hours (sum of sequential last-resort truck routes)
- Combined Total Time: 1432.7 hours
- Number of trucks (main hybrid): 85
- Total Driver Cost: €31749.67 (sum of all driver hours for hybrid and last-resort)

## Financial Analysis

The Netro system saves **1112.20 EUR** in driver costs,
representing a 3.38% reduction compared to the traditional approach.

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
- Truck Distance: 66465.56 km
- Robot Distance: 4693.90 km
- Distance Ratio (Robot/Truck): 0.1:1