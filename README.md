# 3lcvrp-alns-dblf

Goal: implement the hybrid heuristic from the paper: outer Adaptive Large Neighbourhood Search (ALNS) for routing + inner Deepest-Bottom-Left-Fill (DBLF) packing heuristic that checks loading constraints while placing items (C1..C10) to produce feasible packing plans. The code below provides:

- Basic data model (items, customers, vehicle)

- DBLF packing heuristic using "spaces" (front/right/top) as described in the paper

- Constraint checking for geometry (no overlap / inside vehicle), orthogonality/rotation, capacity, LIFO/MLIFO, minimal supporting area (C6a), simplified load-bearing (C7b1, simplified selection), reachability (C8) and axle & balanced checks (C9, C10) in the form presented in the paper

- A minimal ALNS skeleton that uses removal/insertion operators and calls the packing procedure on routes (skeleton only — full tuning left for later)

Paper reference (used while designing algorithm & formulas): [Advanced loading constraints for 3D vehicle routing problems](https://doi.org/10.1007/s00291-021-00645-w)