# packing.py
# DBLF packing & constraint checks (used by ALNS)
# Based on: "Advanced loading constraints for 3D vehicle routing problems" (Krebs et al., 2021).

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math, copy
from itertools import permutations

DEBUG = False  # set True for detailed placement diagnostics


@dataclass
class Item:
    id: str
    mass: float
    l: int
    w: int
    h: int
    fragile: bool = False
    lbs: Optional[float] = None   # load-bearing strength
    customer_id: Optional[int] = None
    pos: Optional[Tuple[int,int,int]] = None
    orientation: Optional[Tuple[int,int,int]] = None


@dataclass
class Customer:
    id: int
    items: List[Item]


@dataclass
class VehicleSpec:
    L: int
    W: int
    H: int
    capacity: float
    Lf: float = 0.5
    WB: float = 3.0
    FAperm: float = 2000.0
    RAperm: float = 4000.0
    balanced_p: float = 0.7


@dataclass
class Space:
    x:int; y:int; z:int; lx:int; wy:int; hz:int

    def contains_item(self, it_l, it_w, it_h) -> bool:
        return (it_l <= self.lx) and (it_w <= self.wy) and (it_h <= self.hz)


def overlaps(a_min, a_max, b_min, b_max):
    return not (a_max <= b_min or b_max <= a_min)


@dataclass
class PackingState:
    vehicle: VehicleSpec
    placed_items: List[Item] = field(default_factory=list)
    spaces: List[Space] = field(default_factory=list)

    def total_mass(self) -> float:
        return sum(it.mass for it in self.placed_items)

    def fits_in_vehicle(self, x, y, z, l, w, h) -> bool:
        V = self.vehicle
        if x < 0 or y < 0 or z < 0: 
            return False
        if x + l > V.L or y + w > V.W or z + h > V.H: 
            return False
        for it in self.placed_items:
            ix, iy, iz = it.pos
            il, iw, ih = it.orientation
            if overlaps(x, x+l, ix, ix+il) and overlaps(y, y+w, iy, iy+iw) and overlaps(z, z+h, iz, iz+ih):
                return False
        return True

    def capacity_ok(self, additional_mass=0.0) -> bool:
        return (self.total_mass() + additional_mass) <= self.vehicle.capacity + 1e-9

    def lifo_ok(self, item: Item, route_customers_order: List[int], allow_mlifo=False):
        if item.customer_id is None:
            return True
        posx = item.pos[0]
        for it in self.placed_items:
            if it.customer_id is None:
                continue
            if route_customers_order.index(it.customer_id) < route_customers_order.index(item.customer_id):
                # The later-served item must not block the earlier one
                if it.pos[0] < posx + item.orientation[0]:
                    if not allow_mlifo:
                        return False
                    else:
                        if it.pos[2] < item.pos[2] + item.orientation[2]:
                            return False
        return True

    def minimal_support_ok(self, item: Item, alpha=0.75):
        x,y,z = item.pos
        l,w,_ = item.orientation
        if z == 0:
            return True
        supported_area = 0.0
        footprint = l * w
        for it in self.placed_items:
            itx, ity, itz = it.pos
            il, iw, ih = it.orientation
            if abs((itz + ih) - z) < 1e-6:
                x_overlap = max(0, min(x+l, itx+il) - max(x, itx))
                y_overlap = max(0, min(y+w, ity+iw) - max(y, ity))
                supported_area += x_overlap * y_overlap
        return (supported_area / (footprint + 1e-12)) >= alpha

    def load_bearing_ok_simplified(self, item: Item):
        x,y,z = item.pos
        l,w,_ = item.orientation
        supports = []
        for it in self.placed_items:
            itx, ity, itz = it.pos
            il, iw, ih = it.orientation
            if abs((itz + ih) - z) < 1e-6:
                x_overlap = max(0, min(x+l, itx+il) - max(x, itx))
                y_overlap = max(0, min(y+w, ity+iw) - max(y, ity))
                area = x_overlap * y_overlap
                if area > 0:
                    supports.append((it, area))
        if not supports:
            return False
        total_support_area = sum(a for _,a in supports)
        for it, area in supports:
            share = (area / total_support_area) * item.mass
            if it.lbs is not None and area > 0:
                added_load_per_area = share / area
                if added_load_per_area > it.lbs + 1e-9:
                    return False
        return True

    def reachability_ok(self, item: Item, max_reach_dm=5):
        x,y,z = item.pos
        l,w,_ = item.orientation
        distance = self.vehicle.L - (x + l)
        return distance <= max_reach_dm + 1e-9

    def axle_weights_ok(self):
        V = self.vehicle
        g = 9.81
        numerator = 0.0
        for it in self.placed_items:
            xi = it.pos[0]
            si = V.Lf + xi + it.orientation[0]/2.0
            numerator += it.mass * g * si
        FRA = numerator / V.WB
        FFA = (sum(it.mass * g for it in self.placed_items)) - FRA
        if FFA > V.FAperm * g + 1e-6:
            return False
        if FRA > V.RAperm * g + 1e-6:
            return False
        if FFA < -1e-6 or FRA < -1e-6:
            return False
        return True

    def balanced_loading_ok(self):
        V = self.vehicle
        left_mass = 0.0
        right_mass = 0.0
        halfw = V.W / 2.0
        for it in self.placed_items:
            yi = it.pos[1]
            wi = it.orientation[1]
            left_portion = max(0.0, min(halfw, yi+wi) - max(0.0, yi)) / (wi + 1e-12)
            right_portion = 1.0 - left_portion
            left_mass += it.mass * left_portion
            right_mass += it.mass * right_portion
        if left_mass > V.capacity * V.balanced_p + 1e-9:
            return False
        if right_mass > V.capacity * V.balanced_p + 1e-9:
            return False
        return True


def split_space(space: Space, item: Item) -> List[Space]:
    new_spaces = []
    sx, sy, sz, slx, swy, shz = space.x, space.y, space.z, space.lx, space.wy, space.hz
    ix, iy, iz = item.pos
    il, iw, ih = item.orientation

    sx_max, sy_max, sz_max = sx + slx, sy + swy, sz + shz
    ix_max, iy_max, iz_max = ix + il, iy + iw, iz + ih

    if (ix >= sx_max or ix_max <= sx or 
        iy >= sy_max or iy_max <= sy or 
        iz >= sz_max or iz_max <= sz):
        return [space]

    # left/right splits
    if ix > sx:
        new_spaces.append(Space(sx, sy, sz, ix - sx, swy, shz))
    if ix_max < sx_max:
        new_spaces.append(Space(ix_max, sy, sz, sx_max - ix_max, swy, shz))

    # front/back splits
    y0 = max(sy, iy)
    y1 = min(sy_max, iy_max)
    if y0 > sy:
        new_spaces.append(Space(sx, sy, sz, slx, y0 - sy, shz))
    if y1 < sy_max:
        new_spaces.append(Space(sx, y1, sz, slx, sy_max - y1, shz))

    # bottom/top splits
    z0 = max(sz, iz)
    z1 = min(sz_max, iz_max)
    if z0 > sz:
        new_spaces.append(Space(sx, sy, sz, slx, swy, z0 - sz))
    if z1 < sz_max:
        new_spaces.append(Space(sx, sy, z1, slx, swy, sz_max - z1))

    return [s for s in new_spaces if s.lx > 0 and s.wy > 0 and s.hz > 0]


def update_spaces_with_item(state: PackingState, item: Item):
    new_spaces = []
    for sp in state.spaces:
        sx, sy, sz, slx, swy, shz = sp.x, sp.y, sp.z, sp.lx, sp.wy, sp.hz
        ix, iy, iz = item.pos
        il, iw, ih = item.orientation

        sx_max, sy_max, sz_max = sx + slx, sy + swy, sz + shz
        ix_max, iy_max, iz_max = ix + il, iy + iw, iz + ih

        if (ix >= sx_max or ix_max <= sx or
            iy >= sy_max or iy_max <= sy or
            iz >= sz_max or iz_max <= sz):
            new_spaces.append(sp)
            continue

        new_spaces.extend(split_space(sp, item))

    state.spaces = [s for s in new_spaces if s.lx > 0 and s.wy > 0 and s.hz > 0]


def _all_orientations(item: Item):
    # YOU REQUESTED ONLY 2 ORIENTATIONS → KEEP EXACTLY THESE TWO
    return [
        (item.l, item.w, item.h),
        (item.w, item.l, item.h)
    ]


def dbllf_pack(route_customers: List[Customer], vehicle: VehicleSpec, loading_constraints: dict):
    state = PackingState(vehicle=vehicle)
    state.spaces = [Space(0,0,0, vehicle.L, vehicle.W, vehicle.H)]

    items_seq = []

    # Build items (deep copies!)
    for cust in reversed(route_customers):
        sorted_items = sorted(cust.items, key=lambda it: (it.fragile, -(it.l*it.w*it.h), -it.l, -it.w))
        for it in sorted_items:
            new_it = copy.deepcopy(it)      # ← critical
            new_it.customer_id = cust.id
            new_it.pos = None
            new_it.orientation = None
            items_seq.append(new_it)

    route_order = [c.id for c in route_customers]

    # Placement loop
    for item in items_seq:
        placed = False

        state.spaces.sort(key=lambda s: (-s.x, s.z, s.y))  # DBL: deepest → bottom → left

        for sp in list(state.spaces):

            for (ol, ow, oh) in _all_orientations(item):

                if not sp.contains_item(ol, ow, oh):
                    continue

                candx, candy, candz = sp.x, sp.y, sp.z

                if not state.fits_in_vehicle(candx, candy, candz, ol, ow, oh):
                    continue

                # Tentative placement
                item.pos = (candx, candy, candz)
                item.orientation = (ol, ow, oh)

                ok = True
                fail = []

                if not state.capacity_ok(additional_mass=item.mass):
                    ok = False; fail.append("capacity")

                if ok and not state.lifo_ok(item, route_order, allow_mlifo=loading_constraints.get("MLIFO", False)):
                    ok = False; fail.append("lifo")

                if ok and not state.minimal_support_ok(item, alpha=loading_constraints.get("min_support", 0.75)):
                    ok = False; fail.append("minimal_support")

                # Additional constraints can be optionally tested:
                # if ok and not state.load_bearing_ok_simplified(item): ok=False; fail.append("load_bearing")

                if not ok and DEBUG:
                    print(f"[PACK DEBUG] FAIL item {item.id} cust={item.customer_id} pos={item.pos} "
                          f"orient={item.orientation} space={sp} reasons={fail}")

                if ok:
                    # commit
                    state.placed_items.append(copy.deepcopy(item))

                    update_spaces_with_item(state, item)

                    # **IMPORTANT:** NO MORE AGGRESSIVE SPACE PRUNING!
                    placed = True
                    break

                item.pos = None
                item.orientation = None

            if placed:
                break

        if not placed:
            if DEBUG:
                print(f"[PACK DEBUG] FAILED TO PLACE ITEM {item.id} (cust {item.customer_id})")
            return None

    return state