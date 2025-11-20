import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# -------------------------
# Data models
# -------------------------
@dataclass
class Item:
    id: str
    mass: float
    l: int
    w: int
    h: int
    fragile: bool = False
    lbs: Optional[float] = None   # load bearing strength (mass per area unit)
    # orientation: (l,w,h) after rotation - filled when placed
    pos: Optional[Tuple[int,int,int]] = None  # x,y,z of lower-left-near corner
    orientation: Optional[Tuple[int,int,int]] = None  # l,w,h after rotation
    customer_id: Optional[int] = None

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
    Lf: float = 0.5  # distance from front axle to loading space (example)
    WB: float = 3.0  # wheelbase (example)
    FAperm: float = 2000.0  # permissible front axle mass (kg)
    RAperm: float = 4000.0  # permissible rear axle mass (kg)
    balanced_p: float = 0.7  # max mass ratio per side vs D

# -------------------------
# Utility: axis-aligned overlap test
# -------------------------
def overlaps(a_min, a_max, b_min, b_max):
    return not (a_max <= b_min or b_max <= a_min)

# -------------------------
# Represent free space as cuboid: (x, y, z, lx, wy, hz)
# -------------------------
@dataclass
class Space:
    x:int; y:int; z:int; lx:int; wy:int; hz:int

    def contains_item(self, it_l, it_w, it_h) -> bool:
        return (it_l <= self.lx) and (it_w <= self.wy) and (it_h <= self.hz)

# -------------------------
# Packing state and checks
# -------------------------
@dataclass
class PackingState:
    vehicle: VehicleSpec
    placed_items: List[Item] = field(default_factory=list)  # with pos and orientation set
    spaces: List[Space] = field(default_factory=list)

    def total_mass(self) -> float:
        return sum(it.mass for it in self.placed_items)

    def fits_in_vehicle(self, x, y, z, l, w, h) -> bool:
        V = self.vehicle
        if x < 0 or y < 0 or z < 0: return False
        if x + l > V.L or y + w > V.W or z + h > V.H: return False
        # overlap check with placed items
        for it in self.placed_items:
            ix, iy, iz = it.pos
            il, iw, ih = it.orientation
            if overlaps(x, x+l, ix, ix+il) and overlaps(y, y+w, iy, iy+iw) and overlaps(z, z+h, iz, iz+ih):
                return False
        return True

    # simple capacity check
    def capacity_ok(self, additional_mass=0.0) -> bool:
        return (self.total_mass() + additional_mass) <= self.vehicle.capacity + 1e-9

    # LIFO / MLIFO: ensure we don't place items above items of customers served later.
    def lifo_ok(self, item: Item, route_customers_order: List[int], allow_mlifo=False):
        # item.customer_id is the customer being placed; we assume items are placed in reversed route order in packing sequence.
        # For LIFO: no item belonging to a later-served customer may be below/behind this item.
        # We'll implement a conservative check: any placed item that is "in front" (smaller x) and belongs to a later customer => violation
        posx = item.pos[0]
        for it in self.placed_items:
            if it.customer_id is None or item.customer_id is None: continue
            # if that placed item belongs to a customer served after item.customer_id (index wise)
            if route_customers_order.index(it.customer_id) > route_customers_order.index(item.customer_id):
                # placed item is served later -> must not be under/in front of current item
                # check whether it is below or overlapping in x-front direction
                if it.pos[0] < posx + item.orientation[0]:  # overlapping in front/back sense (conservative)
                    # MLIFO allows hanging over (i.e., being partly over without touching)
                    # Hard check: if the placed item touches (z below and intersects area) then violation
                    if not allow_mlifo:
                        return False
                    else:
                        # allow hanging: skip only if it's not directly supporting (i.e., not overlapping in z)
                        # We keep conservative implementation: forbid placing fully on top of previously placed later customer
                        if it.pos[2] < item.pos[2] + item.orientation[2]:
                            return False
        return True

    # minimal supporting area (C6a): check that at least ratio alpha of base area is supported by directly underlying items
    def minimal_support_ok(self, item: Item, alpha=0.75):
        # if placed at z=0 -> ground support OK
        x,y,z = item.pos
        l,w,_ = item.orientation
        if z == 0: return True
        # we need to compute supported area: for each placed item directly below (their top face equals z), compute overlapping footprint area
        supported_area = 0.0
        footprint = l * w
        for it in self.placed_items:
            itx, ity, itz = it.pos
            il, iw, ih = it.orientation
            if abs((itz + ih) - z) < 1e-6:  # top face exactly at this height
                # compute area overlap between rectangles in x-y plane
                x_overlap = max(0, min(x+l, itx+il) - max(x, itx))
                y_overlap = max(0, min(y+w, ity+iw) - max(y, ity))
                supported_area += x_overlap * y_overlap
        return (supported_area / footprint) + 1e-9 >= alpha

    # simplified load-bearing (C7b1 simplified selection): compute load per top-face area, compare with lbs of each underlying item
    def load_bearing_ok_simplified(self, item: Item):
        # when item is placed, its full mass is distributed to items directly under its footprint (located under its base)
        x,y,z = item.pos
        l,w,_ = item.orientation
        footprint = l*w
        # collect directly underlying items (their top face at z):
        supports = []
        for it in self.placed_items:
            itx, ity, itz = it.pos
            il, iw, ih = it.orientation
            if abs((itz + ih) - z) < 1e-6:
                # overlap area:
                x_overlap = max(0, min(x+l, itx+il) - max(x, itx))
                y_overlap = max(0, min(y+w, ity+iw) - max(y, ity))
                area = x_overlap * y_overlap
                if area > 0:
                    supports.append((it, area))
        if not supports:
            # unsupported -> failing robust stability / load-bearing (except z==0 handled earlier)
            return False
        # calculate support share per item
        total_support_area = sum(a for _,a in supports)
        # distribute mass proportionally to support area
        for it, area in supports:
            share = (area / total_support_area) * item.mass
            # compare share per area unit vs underlying's lbs (if provided)
            if it.lbs is not None:
                # load per area on the top of the underlying item might be cumulative; we do a conservative check: this additional share/area must not exceed lbs
                added_load_per_area = share / area if area>0 else float('inf')
                if added_load_per_area > it.lbs + 1e-9:
                    return False
        return True

    # reachability: distance from front (door) to the front-most face of the item must be <= max_reach (we assume front at x=vehicle.L)
    def reachability_ok(self, item: Item, max_reach_dm=5):
        # In paper they measure in dm — here we use same units as item dims. We'll compute distance from door (front) to nearest front face
        # In our coord system origin is at deepest-back; so front (door) is at x = 0 (driver's door is at x=vehicle.L)
        # We'll interpret driver standing at rear (x = vehicle.L) and reach towards decreasing x; distance = vehicle.L - (x + l) (distance from rear to front of item)
        x,y,z = item.pos
        l,w,_ = item.orientation
        distance = self.vehicle.L - (x + l)
        return distance <= max_reach_dm + 1e-9

    # axle weight check (C9)
    def axle_weights_ok(self):
        V = self.vehicle
        g = 9.81
        # compute FRA as in paper:
        numerator = 0.0
        for it in self.placed_items:
            xi = it.pos[0]
            si = V.Lf + xi + it.orientation[0]/2.0
            numerator += it.mass * g * si
        FRA = numerator / V.WB
        FFA = (sum(it.mass * g for it in self.placed_items)) - FRA
        # compare
        if FFA > V.FAperm * g + 1e-6: return False
        if FRA > V.RAperm * g + 1e-6: return False
        if FFA < -1e-6 or FRA < -1e-6: return False
        return True

    # balanced loading C10 (simplified per paper formula)
    def balanced_loading_ok(self):
        # compute left and right assigned masses as in formula 12 & 13 (approx)
        V = self.vehicle
        left_mass = 0.0
        right_mass = 0.0
        halfw = V.W / 2.0
        for it in self.placed_items:
            yi = it.pos[1]
            wi = it.orientation[1]
            xi = it.pos[0]
            # mass proportional assignment by projecting onto width dimension (simplified)
            # portion on left side:
            left_portion = max(0.0, min(halfw, yi+wi) - max(0.0, yi)) / wi
            right_portion = 1.0 - left_portion
            left_mass += it.mass * left_portion
            right_mass += it.mass * right_portion
        if left_mass > V.capacity * V.balanced_p + 1e-9: return False
        if right_mass > V.capacity * V.balanced_p + 1e-9: return False
        return True

# -------------------------
# DBLF packing algorithm (simplified but faithful)
# -------------------------
def dbllf_pack(route_customers: List[Customer], vehicle: VehicleSpec, loading_constraints: dict):
    """
    route_customers: customers in the route in visiting order (first visited -> first unloaded)
    We will pack items in reversed visiting order (last visited first in packing sequence) as described in paper.
    Returns PackingState if success, else None
    """
    # Initialize packing state
    state = PackingState(vehicle=vehicle)
    # initial full space (origin at deepest-bottom-left)
    state.spaces = [Space(0,0,0, vehicle.L, vehicle.W, vehicle.H)]
    # create packing sequence: items sorted per customer by fragility, volume, length, width; added reversed to route visiting order
    items_seq = []
    for cust in reversed(route_customers):
        sorted_items = sorted(cust.items, key=lambda it: (it.fragile, -(it.l*it.w*it.h), -it.l, -it.w))
        for it in sorted_items:
            it.customer_id = cust.id
            items_seq.append(it)
    # route_customers_order used for LIFO checks (list of customer ids in visiting order)
    route_order = [c.id for c in route_customers]

    # For each item, try all spaces, all 2 rotations (rotation only in l-w plane) and place according DBL rule
    for item in items_seq:
        placed = False
        # recompute smallest dims to prune spaces (paper uses lmin,hmin — we'll use item dims)
        # sort spaces by DBL: deepest (largest x), then bottom (smallest z), then left (smallest y)? Paper: deepest-back first -> highest x (since origin at back?). In our origin deepest-back-left = (0,0,0),
        # we'll interpret "deepest" = largest x; so sort descending x, ascending z, ascending y.
        state.spaces.sort(key=lambda s: (-s.x, s.z, s.y))
        for sp in list(state.spaces):
            # try both orientations (no rotation about vertical axis beyond swap l/w)
            orientations = [(item.l, item.w, item.h), (item.w, item.l, item.h)] if item.l != item.w else [(item.l,item.w,item.h)]
            for (ol, ow, oh) in orientations:
                if not sp.contains_item(ol,ow,oh): continue
                # candidate position is the deepest-bottom-left in this space -> x=sp.x, y=sp.y, z=sp.z
                candx, candy, candz = sp.x, sp.y, sp.z
                if not state.fits_in_vehicle(candx, candy, candz, ol, ow, oh):
                    continue
                # temporarily set item pos/orientation to test constraints
                item.pos = (candx, candy, candz)
                item.orientation = (ol, ow, oh)

                # check constraints:
                ok = True
                if not state.capacity_ok(additional_mass=item.mass): ok = False
                if ok and not state.lifo_ok(item, route_order, allow_mlifo=loading_constraints.get("MLIFO", False)): ok = False
                if ok and not state.minimal_support_ok(item, alpha=loading_constraints.get("min_support", 0.75)): ok = False
                # if ok and not state.load_bearing_ok_simplified(item): ok = False
                # if ok and not state.reachability_ok(item, max_reach_dm=loading_constraints.get("reachability", 5)): ok = False
                # if ok and not state.axle_weights_ok(): ok = False
                # if ok and not state.balanced_loading_ok(): ok = False

                if ok:
                    # Accept placement: add to placed_items
                    state.placed_items.append(Item(**{**item.__dict__}))  # shallow copy to freeze position
                    placed = True
                    # create front/right/top new spaces as in paper (simplified)
                    # front: x = candx + ol, bounds y same, z same -> length = sp.lx - ol
                    # But to keep code simple and avoid fragmentation explosion, create three candidate spaces:
                    # front space:
                    if sp.lx - ol > 0:
                        new_sp = Space(candx + ol, candy, candz, sp.lx - ol, ow, oh)  # simplified bounds
                        state.spaces.append(new_sp)
                    # right space:
                    if sp.wy - ow > 0:
                        new_sp = Space(candx, candy + ow, candz, ol, sp.wy - ow, oh)
                        state.spaces.append(new_sp)
                    # top space:
                    if sp.hz - oh > 0:
                        new_sp = Space(candx, candy, candz + oh, ol, ow, sp.hz - oh)
                        state.spaces.append(new_sp)
                    # remove used space
                    try:
                        state.spaces.remove(sp)
                    except ValueError:
                        pass
                    # cleanup: remove spaces too small (we will simply filter spaces that cannot contain any remaining items)
                    remaining_min_l = min([it.l for it in items_seq if it.pos is None] + [1])
                    remaining_min_h = min([it.h for it in items_seq if it.pos is None] + [1])
                    state.spaces = [s for s in state.spaces if s.lx >= remaining_min_l and s.hz >= remaining_min_h]
                    break
                else:
                    # reset temporary pos/orientation
                    item.pos = None
                    item.orientation = None
            if placed: break
        if not placed:
            # no feasible position found -> packing fails
            return None
    # After all placed, return state
    return state

# -------------------------
# Small demo / example usage
# -------------------------
def demo():
    # create vehicle
    vehicle = VehicleSpec(L=12, W=2, H=2.5, capacity=2000.0, Lf=0.5, WB=3.0, FAperm=2000, RAperm=4000, balanced_p=0.7)
    # create a tiny route with two customers
    c1_items = [
        Item(id="a1", mass=30, l=2, w=1, h=1, fragile=False, lbs=5000),
        Item(id="a2", mass=20, l=1, w=1, h=1, fragile=True, lbs=1000)
    ]
    c2_items = [
        Item(id="b1", mass=50, l=3, w=1, h=1, fragile=False, lbs=8000),
        Item(id="b2", mass=10, l=1, w=1, h=1, fragile=False, lbs=5000)
    ]
    cust1 = Customer(id=1, items=c1_items)
    cust2 = Customer(id=2, items=c2_items)
    # route visiting order: cust1 then cust2
    route = [cust1, cust2]
    constraints = {"MLIFO": False, "min_support": 0.75, "reachability": 6}
    state = dbllf_pack(route, vehicle, constraints)
    if state:
        print("Packing succeeded. Placements:")
        for it in state.placed_items:
            print(f"Item {it.id} (cust {it.customer_id}) -> pos={it.pos}, orient={it.orientation}, mass={it.mass}")
    else:
        print("Packing failed for this route.")

if __name__ == "__main__":
    demo()