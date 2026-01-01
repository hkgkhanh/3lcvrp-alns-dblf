import csv
import math
from collections import defaultdict

INPUT_CSV = "3l_cvrp_alns/output/results.csv"
EPSILON = 0.05  # 5%


def coefficient_of_variation(values):
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    return std / mean


def main():
    distances = defaultdict(list)

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance = row["instance"]

            if not instance.startswith("3l_cvrp"):
                continue

            distance = float(row["distance"])
            distances[instance].append(distance)

    total_runs = 0
    successful_runs = 0

    print(f"Threshold: {EPSILON * 100:.1f}% of best observed solution\n")

    for instance, dists in sorted(distances.items()):
        best = min(dists)
        threshold = best * (1.0 + EPSILON)

        count_good = sum(1 for d in dists if d <= threshold)

        total_runs += len(dists)
        successful_runs += count_good

        cv = coefficient_of_variation(dists)

        print(
            f"{instance}: "
            f"{count_good}/{len(dists)} | "
            f"CV = {cv*100:.4f}% "
            f"(best = {best:.3f})"
        )

    ratio = successful_runs / total_runs if total_runs > 0 else 0.0

    print("\n========== SUMMARY ==========")
    print(f"Successful runs: {successful_runs}/{total_runs}")
    print(f"Success ratio: {ratio:.4f}")


if __name__ == "__main__":
    main()
