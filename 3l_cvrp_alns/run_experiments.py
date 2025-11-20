# run_experiments.py

import argparse, os, json, time, csv
from instance_loader import load_txt_instance
from alns import alns, Solution, evaluate_solution
from packing import VehicleSpec, Customer, dbllf_pack
from serializer import packing_to_dict
from initial_solution import build_initial_solution


def compute_fill_rate(solution, vehicle: VehicleSpec):
    """
    Compute average volume usage across all vehicles.
    Packed volume = sum(l*w*h) of placed items in each route's packing.
    """
    if not solution.packing:
        return 0.0

    V_total = vehicle.L * vehicle.W * vehicle.H
    if V_total <= 0:
        return 0.0

    fill_rates = []
    for pack in solution.packing:
        if pack is None:
            fill_rates.append(0.0)
            continue

        used_vol = sum(it.orientation[0] * it.orientation[1] * it.orientation[2]
                       for it in pack.placed_items)
        fill_rates.append(used_vol / V_total)

    if not fill_rates:
        return 0.0
    return sum(fill_rates) / len(fill_rates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=str, default="example_instances", help="folder with instances (.txt)")
    parser.add_argument("--out", type=str, default="results", help="output folder")
    parser.add_argument("--time", type=int, default=60, help="time limit (sec) per run")
    parser.add_argument("--iter", type=int, default=5000, help="max iterations")
    parser.add_argument("--repeats", type=int, default=1, help="repeats per instance")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # CSV output path
    csv_path = os.path.join(args.out, "results.csv")
    write_header = not os.path.exists(csv_path)

    # open CSV
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)

    if write_header:
        writer.writerow(["instance", "distance", "vehicle_count", "fill_rate", "time_elapsed"])

    # load all instance files
    instances = []
    for fn in os.listdir(args.instances):
        if "overview" in fn.lower():
            continue
        if fn.lower().endswith(".txt"):
            path = os.path.join(args.instances, fn)
            vehicle, customers, dist_matrix = load_txt_instance(path)
            instances.append((fn, vehicle, customers, dist_matrix))

    # run ALNS on each instance
    for (name, vehicle, customers, dist_matrix) in instances:
        for r in range(args.repeats):
            print(f"Running {name} repeat {r+1}/{args.repeats}")

            t0 = time.time()

            init = build_initial_solution(
                customers,
                vehicle,
                loading_constraints={
                    "MLIFO": False,
                    "min_support": 0.75,
                    "reachability": 6
                },
                dist_matrix=dist_matrix
            )

            best = alns(
                init,
                vehicle,
                loading_constraints={
                    "MLIFO": False,
                    "min_support": 0.75,
                    "reachability": 6
                },
                max_iter=args.iter,
                time_limit=args.time,
                dist_matrix=dist_matrix
            )

            elapsed = time.time() - t0

            distance = evaluate_solution(
                best, vehicle,
                {"MLIFO": False, "min_support": 0.75, "reachability": 6},
                dist_matrix=dist_matrix
            )

            vehicle_count = len(best.routes)
            fill_rate = compute_fill_rate(best, vehicle)

            # Write CSV row
            writer.writerow([name, distance, vehicle_count, fill_rate, elapsed])

            print(f"[DONE] {name}, dist={distance}, vehicles={vehicle_count}, fill={fill_rate:.3f}, time={elapsed:.2f}s")

    csv_file.close()


if __name__ == "__main__":
    main()