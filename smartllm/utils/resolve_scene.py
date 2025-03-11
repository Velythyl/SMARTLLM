import json


def resolve_scene_id(floor_name):
    if isinstance(floor_name, int) or floor_name.startswith("FloorPlan"):
        floor_name = str(floor_name)
        return f"FloorPlan{floor_name.replace('FloorPlan', '')}"

    assert floor_name.endswith(".json")

    with open(floor_name, "r") as f:
        scene = json.load(f)
    return scene