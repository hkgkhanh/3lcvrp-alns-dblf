import os
import re
import csv
import math

INPUT_DIR = "3l_cvrp_alns/output/Gendreau_et_al_2006"
OUTPUT_CSV = "3l_cvrp_alns/output/balance_loading.csv"
OUTPUT_SUMMARY_CSV = "3l_cvrp_alns/output/balance_loading_summary.csv"

CONTAINER_LENGTH = 60
CONTAINER_WIDTH = 25
CONTAINER_HEIGHT = 30

SAFE_RATIO = 0.5


def parse_solution_file(filepath):
    vehicles = []
    current_vehicle = None
    reading_items = False

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Tour_Id:"):
                if current_vehicle is not None:
                    vehicles.append(current_vehicle)
                current_vehicle = []
                reading_items = False
                continue

            if line.startswith("CustId"):
                reading_items = True
                continue

            if line.startswith("----------------------------------------"):
                reading_items = False
                continue

            if reading_items and current_vehicle is not None and line:
                parts = re.split(r"\s+", line)
                if len(parts) < 12:
                    continue

                x = float(parts[4])
                z = float(parts[6])
                length = float(parts[7])
                width = float(parts[8])
                mass = float(parts[10])

                x_center = x + width / 2.0
                z_center = z + length / 2.0

                current_vehicle.append((x_center, z_center, mass))

    if current_vehicle is not None:
        vehicles.append(current_vehicle)

    return vehicles


def compute_vehicle_com(vehicle_items):
    total_mass = sum(m for _, _, m in vehicle_items)
    if total_mass == 0:
        return 0.5, 0.5

    x_c = sum(x * m for x, _, m in vehicle_items) / total_mass
    z_c = sum(z * m for _, z, m in vehicle_items) / total_mass

    return x_c / CONTAINER_WIDTH, z_c / CONTAINER_LENGTH


def compute_vehicle_balance_load(xc, zc):
    dx = abs(xc - 0.5)
    dz = abs(zc - 0.5)

    a = math.sqrt(dx * dx + dz * dz)

    b_candidates = []
    if dx > 0:
        b_candidates.append(0.5 / dx)
    if dz > 0:
        b_candidates.append(0.5 / dz)

    if not b_candidates:
        return 0.0

    b = min(b_candidates)
    return a / b


def is_safe_vehicle(xc, zc, ratio=0.2):
    """
    ratio: float in (0, 1)
           fraction of container size for each dimension
    """
    half = ratio / 2.0
    return (
        0.5 - half <= xc <= 0.5 + half and
        0.5 - half <= zc <= 0.5 + half
    )


def main():
    results = []
    summary = {}  # base_instance -> list of metrics

    for filename in sorted(os.listdir(INPUT_DIR)):
        if not filename.endswith(".txt"):
            continue

        instance = filename.replace(".txt", "")
        filepath = os.path.join(INPUT_DIR, filename)

        # base instance: 3l_cvrp01 from 3l_cvrp01_7
        base_instance = instance.rsplit("_", 1)[0]

        vehicles = parse_solution_file(filepath)

        balance_loads = []
        safe_vehicles = 0

        for v in vehicles:
            xc, zc = compute_vehicle_com(v)
            balance_loads.append(compute_vehicle_balance_load(xc, zc))
            if is_safe_vehicle(xc, zc, SAFE_RATIO):
                safe_vehicles += 1

        total_vehicles = len(vehicles)
        avg_balance_load = (
            sum(balance_loads) / total_vehicles if total_vehicles > 0 else 0.0
        )
        p_safe = safe_vehicles / total_vehicles if total_vehicles > 0 else 0.0

        results.append(
            (instance, avg_balance_load, total_vehicles, safe_vehicles, p_safe)
        )

        if base_instance not in summary:
            summary[base_instance] = []

        summary[base_instance].append(
            (avg_balance_load, total_vehicles, safe_vehicles, p_safe)
        )

    # ===== CSV chi tiết =====
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["instance", "balance_load", "total_vehicles", "safe_vehicles", "p_safe"]
        )
        for row in results:
            writer.writerow(row)

    # ===== CSV trung bình theo instance =====
    with open(OUTPUT_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["instance", "balance_load", "total_vehicles", "safe_vehicles", "p_safe"]
        )

        for base_instance, values in sorted(summary.items()):
            n = len(values)

            mean_balance = sum(v[0] for v in values) / n
            mean_total = sum(v[1] for v in values) / n
            mean_safe = sum(v[2] for v in values) / n
            mean_p_safe = sum(v[3] for v in values) / n

            writer.writerow(
                (base_instance, mean_balance, mean_total, mean_safe, mean_p_safe)
            )

    print(f"Saved detailed results to {OUTPUT_CSV}")
    print(f"Saved summary results to {OUTPUT_SUMMARY_CSV}")



if __name__ == "__main__":
    main()
