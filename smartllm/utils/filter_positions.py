from typing import List
import numpy as np
from SMARTLLM.smartllm.utils.get_obj_of_interest import ObjOfInterest

def is_inside_aabb(agent_pos, obj_pos, obj_size):
    """
    Check if an agent at `agent_pos` is inside the AABB defined by `obj_pos` and `obj_size`.
    """
    for i in range(3):  # Check x, y, z
        if obj_pos[i] - obj_size[i] <= agent_pos[i] <= obj_pos[i] + obj_size[i]:
            return True  # Agent is outside the AABB in at least one dimension
    return False  # Agent is inside the AABB in all dimensions

def _filter_agent_positions(agent_positions, object_positions, object_sizes, margin=0.0):
    """
    Remove agent positions that are inside any object's AABB, considering a margin.
    """
    inflated_sizes = [np.array(size) + margin for size in object_sizes]  # Inflate sizes by margin
    filtered_agents = []

    for agent in agent_positions:
        inside_any_object = False
        for obj, size in zip(object_positions, inflated_sizes):
            if is_inside_aabb(agent, obj, size):
                inside_any_object = True
                break
        if not inside_any_object:
            filtered_agents.append(agent)

    return filtered_agents

def filter_agent_positions(possible_agent_positions, objects_of_interest: List[ObjOfInterest], margin=0.0):
    """
    Filter agent positions to remove those that are inside any object's AABB.
    """
    aabbs = [x.aabb for x in objects_of_interest]
    centers = [x.center for x in aabbs]
    sizes = [x.size for x in aabbs]

    return _filter_agent_positions(possible_agent_positions, centers, sizes, margin)