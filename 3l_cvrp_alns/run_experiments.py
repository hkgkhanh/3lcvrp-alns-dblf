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


def save_solution_txt(
    out_path,
    instance_name,
    solution,
    vehicle_count,
    total_distance,
    elapsed_time
):
    with open(out_path, "w") as f:
        # ===== HEADER =====
        f.write(f"Name:\t\t\t\t{instance_name}\n")
        f.write(f"Problem:\t\t\t3L-CVRP\n")
        f.write(f"Number_of_used_Vehicles:\t{vehicle_count}\n")
        f.write(f"Total_Travel_Distance:\t\t{total_distance}\n")
        f.write(f"Calculation_Time:\t\t{elapsed_time}\n\n")

        # ===== TOURS =====
        for tour_id, route in enumerate(solution.routes):
            pack = solution.packing[tour_id]

            customers = route[:]  # list of customer ids
            no_customers = len(customers)
            no_items = len(pack.placed_items) if pack else 0

            f.write("-" * 40 + "\n")
            f.write(f"Tour_Id:\t\t\t{tour_id}\n")
            f.write(f"No_of_Customers:\t\t{no_customers}\n")
            f.write(f"No_of_Items:\t\t\t{no_items}\n")
            f.write(
                "Customer_Sequence:\t\t"
                + " ".join(str(c.id) for c in customers)
                + "\n\n"
            )

            # Item table header
            f.write(
                "CustId\tId\tTypeId\tRotated\tx\ty\tz\t"
                "Length\tWidth\tHeight\tmass\tFragility\n"
            )

            if not pack:
                continue

            for it in pack.placed_items:
                x, y, z = it.pos
                l, w, h = it.orientation

                f.write(
                    f"{it.customer_id}\t"
                    f"{it.id}\t"
                    f"{it.type_id}\t"
                    f"{1 if it.rotated else 0}\t"
                    f"{x}\t{y}\t{z}\t"
                    f"{l}\t{w}\t{h}\t"
                    f"{it.mass}\t"
                    f"{1 if it.fragile else 0}\n"
                )

            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=str, default="example_instances", help="folder with instances (.txt)")
    parser.add_argument("--out", type=str, default="results", help="output folder")
    parser.add_argument("--time", type=int, default=60, help="time limit (sec) per run")
    parser.add_argument("--iter", type=int, default=100, help="max iterations")
    parser.add_argument("--repeats", type=int, default=1, help="repeats per instance")
    parser.add_argument("--start-instance", type=str, default=None, help="resume from this instance filename (e.g. 3l_cvrp07.txt)")
    parser.add_argument("--start-repeat", type=int, default=0, help="resume from this repeat index (0-based)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # CSV output path
    # csv_path = os.path.join(args.out, "results.csv")
    csv_path = "3l_cvrp_alns/output/results.csv"
    write_header = not os.path.exists(csv_path)

    # open CSV
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)

    if write_header:
        writer.writerow(["instance", "distance", "vehicle_count", "fill_rate", "time_elapsed"])

    # load all instance files
    instances = []
    for fn in sorted(os.listdir(args.instances)):
        if "overview" in fn.lower():
            continue
        if fn.lower().endswith(".txt"):
            path = os.path.join(args.instances, fn)
            vehicle, customers, dist_matrix = load_txt_instance(path)
            instances.append((fn, vehicle, customers, dist_matrix))

    if args.start_instance is not None:
        instances = [
            inst for inst in instances
            if inst[0] >= args.start_instance
        ]

    # run ALNS on each instance
    for (name, vehicle, customers, dist_matrix) in instances:
        for r in range(args.repeats):
            # skip repeats if resuming
            if args.start_instance == name and r < args.start_repeat:
                continue
            
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


            # write solution to txt file
            sol_dir = os.path.join(args.out)
            os.makedirs(sol_dir, exist_ok=True)

            sol_name = f"{os.path.splitext(name)[0]}_{r}.txt"
            sol_path = os.path.join(sol_dir, sol_name)

            save_solution_txt(
                out_path=sol_path,
                instance_name=os.path.splitext(name)[0],
                solution=best,
                vehicle_count=vehicle_count,
                total_distance=distance,
                elapsed_time=elapsed
            )

            # Write CSV row
            writer.writerow([name, distance, vehicle_count, fill_rate, elapsed])

            print(f"[DONE] {name}, dist={distance}, vehicles={vehicle_count}, fill={fill_rate:.3f}, time={elapsed:.2f}s")

    csv_file.close()


if __name__ == "__main__":
    main()