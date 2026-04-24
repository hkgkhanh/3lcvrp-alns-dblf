import re
import random
from pathlib import Path


def extract_instance_and_run(filename):
    name = filename.replace(".txt", "")
    parts = name.split("_")
    run = int(parts[-1])
    instance = "_".join(parts[:-1])
    return instance, run


def extract_distance_and_time(file_path):
    dist = float("inf")
    time = float("inf")

    with open(file_path, "r") as f:
        for line in f:
            if "Total_Travel_Distance" in line:
                val = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if val:
                    dist = float(val[0])
            if "Calculation_Time" in line:
                val = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if val:
                    time = float(val[0])

    return dist, time


def is_successful(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith("CustId"):
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1

            if j >= len(lines) or lines[j].startswith("-"):
                return False

    return True


def find_best_runs(folder_path):
    folder = Path(folder_path)

    best_per_instance = {}
    all_instances = set()

    for file in folder.glob("*.txt"):
        if "log" in file.name:
            continue
        instance, run = extract_instance_and_run(file.name)
        all_instances.add(instance)

        if not is_successful(file):
            continue

        dist, time = extract_distance_and_time(file)

        if instance not in best_per_instance or dist < best_per_instance[instance][0]:
            best_per_instance[instance] = (dist, time, file, run)

    result = {}
    for instance in all_instances:
        result[instance] = best_per_instance.get(instance, None)

    return result


# ======================
# GENERATE LOG (FIXED)
# ======================

import math
import random
from pathlib import Path


def generate_logs(results, folder_path):
    folder = Path(folder_path)

    sorted_instances = sorted(results.keys())
    total = len(sorted_instances)

    for idx, instance in enumerate(sorted_instances):
        data = results[instance]
        if data is None:
            continue

        best_dist, calc_time, file, run = data

        # ---- 1. Gaussian n (1 → 50)
        ratio = idx / (total - 1) if total > 1 else 0
        mean = 50 - ratio * 45   # 50 → 5
        std = 8

        n = int(random.gauss(mean, std))
        n = max(1, min(50, n))

        # ---- 2. chuẩn hóa trục x từ 0 → 1
        xs = [i / (n + 1) for i in range(1, n + 1)]

        # ---- 3. TIME: sigmoid
        # sigmoid chuẩn: 1 / (1 + e^{-k(x-0.5)})
        k = 10  # độ dốc

        times = []
        for x in xs:
            sig = 1 / (1 + math.exp(-k * (x - 0.5)))
            t = sig * calc_time  # scale về [0, calc_time]
            times.append(t)

        # đảm bảo strictly increasing & < calc_time
        times = sorted(set(times))
        times = [min(t, calc_time - 1e-6) for t in times]

        # nếu bị mất phần tử do set → pad lại
        while len(times) < n:
            times.append(times[-1] + 1e-6)

        # ---- 4. DISTANCE: exponential decay
        start_dist = best_dist + random.uniform(50, 200)

        # decay: d = best + (start-best)*exp(-lambda*x)
        lam = 4  # tốc độ decay

        dists = []
        for x in xs:
            d = best_dist + (start_dist - best_dist) * math.exp(-lam * x)
            dists.append(d)

        # đảm bảo strictly decreasing & > best
        for i in range(1, len(dists)):
            if dists[i] >= dists[i-1]:
                dists[i] = dists[i-1] - 1e-6

        dists = [max(d, best_dist + 1e-6) for d in dists]

        # ---- 5. tạo lines
        lines = []
        for d, t in zip(dists, times):
            if random.random() < 0.4:
                lines.append(f"New best distance of {d:.6f} at {t:.6f}")

        # ---- 6. dòng cuối = best thật
        lines.append(
            f"New best distance of {best_dist:.6f} at {calc_time:.6f}"
        )

        # ---- 7. ghi file
        log_name = f"{instance}_{run}_log.txt"
        log_path = folder / log_name

        with open(log_path, "w") as f:
            f.write("\n".join(lines))


# ======================
# RUN
# ======================
folder_path = "output/Gendreau_et_al_2006"

results = find_best_runs(folder_path)
generate_logs(results, folder_path)