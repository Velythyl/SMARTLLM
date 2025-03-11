import dataclasses
from typing import List

import numpy as np


@dataclasses.dataclass
class AABB:
    center: np.ndarray
    size: np.ndarray

@dataclasses.dataclass
class ObjOfInterest:
    name: str
    aabb: AABB

def xyzdict_to_np(dico):
    return np.array([dico["x"], dico["y"], dico["z"]])

def get_obj_of_interest(scene, controller) -> List[ObjOfInterest]:
    objs = list([obj["objectId"] for obj in controller.last_event.metadata["objects"]])
    objs_center = list([obj["axisAlignedBoundingBox"] for obj in controller.last_event.metadata["objects"]])

    ret = []
    for i, obj in enumerate(objs):
        found_obj = False
        for scene_obj in scene["objects"]:
            if scene_obj["id"] == obj:
                found_obj = True
                break

        if found_obj:
            aabb = AABB(
                center=xyzdict_to_np(objs_center[i]["center"]), size=xyzdict_to_np(objs_center[i]["size"]))
            objofinterest = ObjOfInterest(name=obj, aabb=aabb)
            ret.append(objofinterest)

    return ret