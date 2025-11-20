# alns_paper_patch.py
# ALNS implementation patched to match Krebs et al. (2021) hyperparameters exactly.
# Paper file (local): /mnt/data/s00291-021-00645-w.pdf
# Use this module as a drop-in replacement for your existing alns.py

from typing import List, Tuple, Dict, Callable
import random, math, time, copy
from packing import Item, Customer, VehicleSpec, dbllf_pack

PAPER_URL = '/mnt/data/s00291-021-00645-w.pdf'

# ---------------------------
# Solution container
# ---------------------------
class Solution:
    def __init__(self, routes: List[List[Customer]]):
        self.routes = routes
        self.packing = [None] * len(routes)
        self.cost = None

# ---------------------------
# Evaluation
# ---------------------------
def evaluate_solution(sol: Solution, vehicle: VehicleSpec, loading_constraints: dict, dist_matrix=None):
    """
    Evaluate solution and fill sol.packing.
    If any route is infeasible (dbllf_pack returns None) -> sol.cost = +inf
    score chosen = total distance (or route length proxy if no dist_matrix)
    """
    total_dist = 0.0
    sol.packing = [None] * len(sol.routes)

    for i, route in enumerate(sol.routes):
        if not route:
            sol.packing[i] = None
            continue
        pp = dbllf_pack(route, vehicle, loading_constraints)
        sol.packing[i] = pp
        if pp is None:
            sol.cost = float("inf")
            return sol.cost
        if dist_matrix:
            prev = 0
            rdist = 0.0
            for c in route:
                try:
                    rdist += dist_matrix[prev][c.id]
                except Exception:
                    rdist += 1.0
                prev = c.id
            try:
                rdist += dist_matrix[prev][0]
            except Exception:
                rdist += 1.0
            total_dist += rdist
        else:
            total_dist += len(route)

    sol.cost = total_dist
    return total_dist

# ---------------------------
# Removal heuristics
# ---------------------------

def random_removal(sol: Solution, nrem: int):
    routes = copy.deepcopy(sol.routes)
    removed = []
    all_pairs = [(ri, ci) for ri, r in enumerate(routes) for ci in range(len(r))]
    random.shuffle(all_pairs)
    for (ri, ci) in all_pairs:
        if len(removed) >= nrem:
            break
        if ri < len(routes) and ci < len(routes[ri]):
            try:
                removed.append(routes[ri].pop(ci))
            except Exception:
                # safe fallback
                for rr in routes:
                    if rr:
                        removed.append(rr.pop(0))
                        break
    routes = [r for r in routes if r]
    return routes, removed


def worst_removal(sol: Solution, nrem: int, vehicle: VehicleSpec, loading_constraints: dict):
    routes = copy.deepcopy(sol.routes)
    cust_scores = []
    for ri, r in enumerate(routes):
        for c in r:
            v = sum((it.l * it.w * it.h) for it in c.items)
            cust_scores.append((ri, c, len(r) + 1e-6 * v))
    cust_scores.sort(key=lambda x: -x[2])
    removed = []
    for (ri, c, _) in cust_scores[:nrem]:
        if c in routes[ri]:
            routes[ri].remove(c)
            removed.append(c)
    routes = [r for r in routes if r]
    return routes, removed


def shaw_removal(sol: Solution, nrem: int, rng=1.0):
    routes = copy.deepcopy(sol.routes)
    all_customers = [c for r in routes for c in r]
    if not all_customers:
        return routes, []
    seed = random.choice(all_customers)
    removed = [seed]

    def remove_by_id(routes_, cid):
        for r in routes_:
            for it in list(r):
                if it.id == cid.id:
                    r.remove(it)
                    return True
        return False

    remove_by_id(routes, seed)

    def relatedness(a: Customer, b: Customer):
        ad = sum(it.l * it.w * it.h for it in a.items)
        bd = sum(it.l * it.w * it.h for it in b.items)
        vol_diff = abs(ad - bd)
        return vol_diff + abs(a.id - b.id) * 0.1

    pool = sorted([c for c in all_customers if c.id != seed.id], key=lambda x: relatedness(seed, x))
    idx = 0
    while len(removed) < nrem and idx < len(pool):
        removed.append(pool[idx])
        remove_by_id(routes, pool[idx])
        idx += 1
    routes = [r for r in routes if r]
    return routes, removed

# ---------------------------
# Insertion helpers (try all positions)
# ---------------------------

def _try_all_positions_and_select(routes: List[List[Customer]], cust: Customer,
                                  vehicle: VehicleSpec, loading_constraints: dict,
                                  dist_matrix=None):
    best = None
    best_routes = None

    if not routes:
        cand = [[cust]]
        tmp = Solution(cand)
        if evaluate_solution(tmp, vehicle, loading_constraints, dist_matrix) != float('inf'):
            return cand
        else:
            return None

    for ri in range(len(routes)):
        for pos in range(len(routes[ri]) + 1):
            cand = copy.deepcopy(routes)
            cand[ri].insert(pos, cust)
            tmp = Solution(cand)
            cost = evaluate_solution(tmp, vehicle, loading_constraints, dist_matrix)
            if cost == float('inf'):
                continue
            if best is None or cost < best:
                best = cost
                best_routes = cand

    cand = copy.deepcopy(routes)
    cand.append([cust])
    tmp = Solution(cand)
    cost = evaluate_solution(tmp, vehicle, loading_constraints, dist_matrix)
    if cost != float('inf') and (best is None or cost < best):
        best = cost
        best_routes = cand

    return best_routes


def greedy_insert(routes: List[List[Customer]], removed: List[Customer],
                  vehicle: VehicleSpec, loading_constraints: dict, dist_matrix=None):
    routes = copy.deepcopy(routes)
    for c in removed:
        best_routes = _try_all_positions_and_select(routes, c, vehicle, loading_constraints, dist_matrix)
        if best_routes is not None:
            routes = best_routes
        else:
            routes.append([c])
    return routes


def regret_k_insert(routes: List[List[Customer]], removed: List[Customer],
                    vehicle: VehicleSpec, loading_constraints: dict, k=2, dist_matrix=None):
    routes = copy.deepcopy(routes)
    remaining = list(removed)
    while remaining:
        regrets = []
        for c in remaining:
            candidates = []
            for ri in range(len(routes)):
                for pos in range(len(routes[ri]) + 1):
                    cand = copy.deepcopy(routes)
                    cand[ri].insert(pos, c)
                    cost = evaluate_solution(Solution(cand), vehicle, loading_constraints, dist_matrix)
                    candidates.append((cost, cand))
            cand = copy.deepcopy(routes)
            cand.append([c])
            cost = evaluate_solution(Solution(cand), vehicle, loading_constraints, dist_matrix)
            candidates.append((cost, cand))

            candidates.sort(key=lambda x: x[0])
            finite = [p for p in candidates if p[0] != float('inf')]
            if not finite:
                regret = 0.0
            else:
                vals = [p[0] for p in finite]
                if len(vals) >= k:
                    regret = vals[k-1] - vals[0]
                else:
                    regret = vals[-1] - vals[0]
            regrets.append((regret, c, candidates))
        regrets.sort(key=lambda x: -x[0])
        _, selc, candidates = regrets[0]
        feasible = [p for p in candidates if p[0] != float('inf')]
        if feasible:
            feasible.sort(key=lambda x: x[0])
            best_cost, best_routes = feasible[0]
            routes = best_routes
        else:
            routes.append([selc])
        remaining.remove(selc)
    return routes

# ---------------------------
# ALNS main (paper hyperparameters applied)
# ---------------------------
def alns(initial_solution: Solution,
         vehicle: VehicleSpec,
         loading_constraints: dict,
         dist_matrix=None,
         max_iter=10000,
         time_limit=None,
         iter_p=100,
         r=0.2,
         restart_no_improve=800,
         verbose=True,
         seed=None):
    # paper hyperparameters
    lambda_cool = 0.9995  # geometric cooling
    sigma_best = 33.0
    sigma_impr = 9.0
    sigma_accept = 3.0
    q_min = 3
    q_max = 33
    k_regret = 2

    if seed is not None:
        random.seed(seed)

    removal_ops = [("random", random_removal),
                   ("worst", lambda s, n: worst_removal(s, n, vehicle, loading_constraints)),
                   ("shaw", shaw_removal)]
    insertion_ops = [("greedy", greedy_insert),
                     ("regret2", lambda routes, removed, v, lc, dm: regret_k_insert(routes, removed, v, lc, k=k_regret, dist_matrix=dm))]

    rem_weights = {name: 1.0 for name, _ in removal_ops}
    ins_weights = {name: 1.0 for name, _ in insertion_ops}
    rem_scores = {name: 0.0 for name, _ in removal_ops}
    ins_scores = {name: 0.0 for name, _ in insertion_ops}
    rem_uses = {name: 1 for name, _ in removal_ops}
    ins_uses = {name: 1 for name, _ in insertion_ops}

    curr = copy.deepcopy(initial_solution)
    evaluate_solution(curr, vehicle, loading_constraints, dist_matrix)
    sbest = copy.deepcopy(curr)
    if curr.cost == float('inf') and verbose:
        print("[ALNS] WARNING: initial solution infeasible (cost=inf). ALNS will attempt feasible neighbors.")

    # initial temperature T0 = 0.1 * initial solution cost (paper)
    initial_cost_for_T = curr.cost if (curr.cost is not None and curr.cost != float('inf')) else 1.0
    T = 0.1 * initial_cost_for_T

    start_time = time.time()
    last_update_iter = 0

    for it in range(1, max_iter + 1):
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print("[ALNS] Time limit reached.")
            break

        if verbose and it % max(1, max_iter // 10) == 0:
            print(f"[ALNS] iter {it} curr={curr.cost:.6g} best={sbest.cost:.6g} T={T:.6g}")

        def select(weights):
            s = sum(weights.values())
            r = random.random() * s
            acc = 0.0
            for k, w in weights.items():
                acc += w
                if r <= acc:
                    return k
            return list(weights.keys())[0]

        rem_name = select(rem_weights)
        ins_name = select(ins_weights)
        rem_func = dict(removal_ops)[rem_name]
        ins_func = dict(insertion_ops)[ins_name]

        Ncust = sum(len(r) for r in curr.routes)
        # sample nrem uniformly in [q_min, q_max] as in the paper, but bounded by Ncust
        nrem = random.randint(q_min, q_max)
        nrem = min(nrem, max(1, Ncust))

        if rem_name == "worst":
            routes_after_rem, removed = rem_func(curr, nrem)
        else:
            routes_after_rem, removed = rem_func(curr, nrem)

        if verbose:
            removed_ids = [getattr(c, "id", None) for c in removed]
            print(f"  operators: rem={rem_name}, ins={ins_name}, nrem={len(removed)}, removed={removed_ids}")

        routes_after_ins = ins_func(routes_after_rem, removed, vehicle, loading_constraints, dist_matrix)

        if routes_after_ins is None:
            if verbose:
                print("  insertion returned None -> skip")
            rem_uses[rem_name] += 1
            ins_uses[ins_name] += 1
            # cooling step
            T *= lambda_cool
            continue

        cand = Solution(routes_after_ins)
        cost = evaluate_solution(cand, vehicle, loading_constraints, dist_matrix)

        if cost == float('inf'):
            if verbose:
                print("  candidate infeasible -> skip")
            rem_uses[rem_name] += 1
            ins_uses[ins_name] += 1
            T *= lambda_cool
            continue

        # acceptance rule (SA) with tolerance
        tol = 1e-9
        accepted = False
        improved = False
        if curr.cost is None:
            curr.cost = float('inf')

        if cost < curr.cost - tol:
            accepted = True
            improved = True
            reason = "improvement"
        elif cost > curr.cost + tol:
            if T > 1e-300:
                prob = math.exp((curr.cost - cost) / T)
            else:
                prob = 0.0
            if random.random() < prob:
                accepted = True
                reason = f"sa(p={prob:.6g})"
            else:
                reason = f"reject(sa p={prob:.6g})"
        else:
            # equal within tol: accept with tiny prob (paper does not emphasize equal handling)
            if random.random() < 0.01:
                accepted = True
                reason = "equal_accept(0.01)"
            else:
                reason = "equal_reject"

        if accepted:
            curr = copy.deepcopy(cand)
            curr.cost = cost
            if verbose:
                print(f"  accepted ({reason}) new_curr={curr.cost:.6g}")
        else:
            if verbose:
                print(f"  rejected ({reason}) cand={cost:.6g} curr={curr.cost:.6g}")

        reward_rem = 0.0
        reward_ins = 0.0
        if curr.cost < sbest.cost - 1e-9:
            reward_rem += sigma_best
            reward_ins += sigma_best
            sbest = copy.deepcopy(curr)
            last_update_iter = it
            if verbose:
                print(f"  *** new global best: {sbest.cost:.6g} at iter {it}")
        elif improved:
            reward_rem += sigma_impr
            reward_ins += sigma_impr
        elif accepted:
            reward_rem += sigma_accept
            reward_ins += sigma_accept

        rem_scores[rem_name] += reward_rem
        rem_uses[rem_name] += 1
        ins_scores[ins_name] += reward_ins
        ins_uses[ins_name] += 1

        if it % iter_p == 0:
            for name in rem_weights.keys():
                if rem_uses[name] > 0:
                    new_w = (1 - r) * rem_weights[name] + r * (rem_scores[name] / rem_uses[name])
                    rem_weights[name] = max(0.1, new_w)
                rem_scores[name] = 0.0
                rem_uses[name] = 1
            for name in ins_weights.keys():
                if ins_uses[name] > 0:
                    new_w = (1 - r) * ins_weights[name] + r * (ins_scores[name] / ins_uses[name])
                    ins_weights[name] = max(0.1, new_w)
                ins_scores[name] = 0.0
                ins_uses[name] = 1
            if verbose:
                print(f"  adapted weights rem={rem_weights} ins={ins_weights}")

        # cooling
        T *= lambda_cool

        if it - last_update_iter > restart_no_improve:
            if verbose:
                print(f"  restart to best at iter {it} (no improve for {restart_no_improve})")
            curr = copy.deepcopy(sbest)

    if verbose:
        print(f"[ALNS] finished. best_cost={sbest.cost:.6g}, vehicles_used={len([r for r in sbest.routes if r])}")
    return sbest