# serializer.py

def item_to_dict(item):
    return {
        "id": item.id,
        "mass": item.mass,
        "l": item.l,
        "w": item.w,
        "h": item.h,
        "fragile": item.fragile,
        "lbs": item.lbs,
        "customer_id": item.customer_id,
        "pos": item.pos,
        "orientation": item.orientation,
    }


def space_to_dict(space):
    # Generic serialization - extend based on your Space structure
    d = {}
    for k, v in vars(space).items():
        try:
            d[k] = v
        except:
            d[k] = str(v)
    return d


def packing_to_dict(packing_state):
    if packing_state is None:
        return None

    return {
        "vehicle": {
            "L": packing_state.vehicle.L,
            "W": packing_state.vehicle.W,
            "H": packing_state.vehicle.H,
            "capacity": packing_state.vehicle.capacity,
            "Lf": packing_state.vehicle.Lf,
            "WB": packing_state.vehicle.WB,
            "FAperm": packing_state.vehicle.FAperm,
            "RAperm": packing_state.vehicle.RAperm,
        },
        "placed_items": [item_to_dict(i) for i in packing_state.placed_items],
        "spaces": [space_to_dict(s) for s in packing_state.spaces],
    }