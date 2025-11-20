# initial_solution.py
# Initial solution builder following Krebs et al. (2021)
# "Advanced loading constraints for 3D vehicle routing problems"

import random
import copy
from typing import List
from packing import Customer, VehicleSpec, dbllf_pack
from alns import Solution, evaluate_solution


def build_initial_solution(customers: List[Customer],
                           vehicle: VehicleSpec,
                           loading_constraints: dict,
                           dist_matrix=None) -> Solution:
    """
    Build the initial solution as described in the paper:
    - Start with an empty solution.
    - Take customers in random order.
    - For each customer:
        * Try inserting at the END of each existing route (LIFO preserving).
        * Feasibility is checked with DBLLF (strict constraints).
        * If feasible, evaluate routing cost.
        * Choose the best feasible insertion.
        * If none feasible -> new route with this customer.
    - Return feasible Solution with all packings computed.

    This matches Section 4.1 (Initial Solution) in Krebs et al. (2021).
    """

    # Start with empty set of routes
    routes: List[List[Customer]] = []

    # Random order, as recommended by ALNS frameworks
    shuffled = customers[:]
    random.shuffle(shuffled)

    for cust in shuffled:
        best_cost = float('inf')
        best_routes = None

        # Try inserting at end of each route (LIFO constraint preserved)
        for ri, r in enumerate(routes):
            # Create candidate by appending customer
            cand_routes = copy.deepcopy(routes)
            cand_routes[ri] = cand_routes[ri] + [cust]

            # Check packing feasibility
            tmp = Solution(cand_routes)
            cost = evaluate_solution(tmp, vehicle, loading_constraints, dist_matrix)
            if cost < best_cost:
                best_cost = cost
                best_routes = cand_routes

        # Option: create a new route if no insertion feasible
        cand_routes = copy.deepcopy(routes)
        cand_routes.append([cust])
        tmp = Solution(cand_routes)
        cost = evaluate_solution(tmp, vehicle, loading_constraints, dist_matrix)
        if cost < best_cost:
            best_cost = cost
            best_routes = cand_routes

        # If best_routes is still None → packing impossible (rare).
        if best_routes is None:
            raise RuntimeError(
                f"Initial solution cannot insert customer {cust.id} "
                "even as a single-customer route."
            )

        routes = best_routes

    # Build final solution
    sol = Solution(routes)
    sol.cost = evaluate_solution(sol, vehicle, loading_constraints, dist_matrix)
    return sol