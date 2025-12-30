# instance_loader.py
# Loader for TXT instances like the one user provided.

import re
from packing import Item, Customer, VehicleSpec


def load_txt_instance(path: str):
    """
    Loads a TXT instance in the exact format the user provided.
    Returns: (vehicle, customers, dist_matrix)
    """

    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # ---------------------------
    # STEP 1 — split sections
    # ---------------------------
    def find(section_name):
        for i, line in enumerate(lines):
            if line.startswith(section_name):
                return i
        raise ValueError(f"Section '{section_name}' not found")

    idx_vehicle = find("VEHICLE")
    idx_customers = find("CUSTOMERS")
    idx_items = find("ITEMS")
    idx_demand = find("DEMANDS PER CUSTOMER")

    vehicle_lines = lines[idx_vehicle + 1 : idx_customers]
    customer_lines = lines[idx_customers + 2 : idx_items]   # skip header
    item_lines = lines[idx_items + 2 : idx_demand]
    demand_lines = lines[idx_demand + 2 :]

    # ---------------------------
    # STEP 2 — parse VEHICLE section
    # ---------------------------
    def parse_vehicle_field(name):
        for line in vehicle_lines:
            if line.startswith(name):
                return float(line.split()[-1])
        # raise ValueError(f"Vehicle field '{name}' not found")
        return -1

    capacity = parse_vehicle_field("Mass_Capacity")
    L = parse_vehicle_field("CargoSpace_Length")
    W = parse_vehicle_field("CargoSpace_Width")
    H = parse_vehicle_field("CargoSpace_Height")
    WB = parse_vehicle_field("Wheelbase")
    max_FA = parse_vehicle_field("Max_Mass_FrontAxle")
    max_RA = parse_vehicle_field("Max_Mass_RearAxle")
    Lf = parse_vehicle_field("Distance_FrontAxle_CargoSpace")

    vehicle = VehicleSpec(
        L=int(L),
        W=int(W),
        H=int(H),
        capacity=float(capacity),
        Lf=float(Lf),
        WB=float(WB),
        FAperm=float(max_FA),
        RAperm=float(max_RA),
    )

    # ---------------------------
    # STEP 3 — parse ITEMS section
    # ---------------------------
    item_types = {}  # Bt1 → Item specification (l,w,h,mass,fragile,lbs)

    for line in item_lines:
        parts = line.split()
        if len(parts) < 7:
            continue

        t = parts[0]
        L_, W_, H_ = map(int, parts[1:4])
        mass = float(parts[4])
        fragile = bool(int(parts[5]))
        lbs = float(parts[6])

        item_types[t] = dict(
            l=L_, w=W_, h=H_, mass=mass, fragile=fragile, lbs=lbs
        )

    # ---------------------------
    # STEP 4 — parse CUSTOMERS section
    # ---------------------------
    customer_data = {}  # id → (x, y)
    for line in customer_lines:
        parts = line.split()
        if len(parts) < 9:
            continue
        cid = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        customer_data[cid] = dict(x=x, y=y)

    # ---------------------------
    # STEP 5 — parse DEMANDS PER CUSTOMER
    # ---------------------------
    customers = []
    cur_customer = None

    for line in demand_lines:

        parts = line.split()
        if len(parts) == 0:
            continue

        if re.match(r"^\d+$", parts[0]):  # new customer group
            cid = int(parts[0])
            cur_customer = Customer(id=cid, items=[])
            customers.append(cur_customer)
            parts = parts[1:]

        # pairs like: Bt3 1  Bt4 1
        i = 0
        while i < len(parts):
            t = parts[i]
            qty = int(parts[i+1])
            i += 2

            if qty > 0:
                spec = item_types[t]
                for k in range(qty):
                    item = Item(
                        id=f"{t}_{cid}_{k}",
                        type_id=f"{t}",
                        rotated=False,
                        mass=spec["mass"],
                        l=spec["l"],
                        w=spec["w"],
                        h=spec["h"],
                        fragile=spec["fragile"],
                        lbs=spec["lbs"],
                    )
                    item.customer_id = cid
                    cur_customer.items.append(item)

    # ---------------------------
    # STEP 6 — distance matrix from coordinates
    # ---------------------------
    # depot assumed to be customer 0
    ids = sorted(customer_data.keys())
    N = len(ids)
    dist_matrix = [[0.0] * (N) for _ in range(N)]

    for i in ids:
        for j in ids:
            xi, yi = customer_data[i]["x"], customer_data[i]["y"]
            xj, yj = customer_data[j]["x"], customer_data[j]["y"]
            dist_matrix[i][j] = ((xi - xj)**2 + (yi - yj)**2) ** 0.5

    return vehicle, customers, dist_matrix