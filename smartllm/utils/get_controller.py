import ai2thor.controller

from hippo.simulation.spatialutils.filter_positions import _filter_agent_positions

from ai2holodeck.constants import THOR_COMMIT_ID
from hippo.ai2thor_hippo_controller import get_hippo_controller_OLDNOW, get_hippo_controller
from hippo.utils.file_utils import get_tmp_folder

TARGET_TMP_DIR = get_tmp_folder()

def get_controller(scene, get_runtime_container=False, **kwargs):
    if "target_dir" in kwargs:
        target_dir = kwargs.pop("target_dir")
    else:
        target_dir = TARGET_TMP_DIR
    return get_hippo_controller_OLDNOW(scene, target_dir=target_dir, get_runtime_container=get_runtime_container, **kwargs)

    if isinstance(scene, str):
        controller = ai2thor.controller.Controller(commit_id=THOR_COMMIT_ID, scene=scene, **kwargs)
    else:
        assert isinstance(scene, dict)
        if "target_dir" in kwargs:
            target_dir = kwargs.pop("target_dir")
        else:
            target_dir = TARGET_TMP_DIR
        controller = get_hippo_controller_OLDNOW(scene, target_dir=target_dir, **kwargs)
    return controller

def get_sim(floor_no):
    from SMARTLLM.smartllm.utils.resolve_scene import resolve_scene_id
    scene = resolve_scene_id(floor_no)

    c, runtime_container = get_hippo_controller(scene, get_runtime_container=True, width=1000, height=1000,
                                                snapToGrid=False, visibilityDistance=100, fieldOfView=90, gridSize=0.25,
                                                rotateStepDegrees=20)

    #c, runtime_container = get_controller(scene, get_runtime_container=True, width=1000, height=1000,
    #                                      snapToGrid=False, visibilityDistance=100, fieldOfView=90, gridSize=0.25,
    #                                      rotateStepDegrees=20)
    no_robot = 1  # len(robots)

    DEFAULT_ROBOT_HEIGHT = 0.95
    DEFAULT_ROBOT_ROT = 90

    c.step(
        action="TeleportFull",
        position={
            "x": 1,
            "y": DEFAULT_ROBOT_HEIGHT, #scene["metadata"]["agent"]["position"]["y"],
            "z": 1,
        },
        rotation=DEFAULT_ROBOT_ROT, #scene["metadata"]["agent"]["rotation"],
        standing=True,
        horizon=30,
        forceAction=True,
    )

    reachable_positions_ = c.step(action="GetReachablePositions").metadata["actionReturn"]
    reachable_positions = positions_tuple = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]

    centers = [obj.position for obj in  runtime_container.objects_map.values()]
    sizes = [obj.size for obj in  runtime_container.objects_map.values()]
    reachable_positions = _filter_agent_positions(reachable_positions, centers, sizes, margin=0.01)

    # initialize n agents into the scene
    c.step(
        dict(action='Initialize', agentMode="default", snapGrid=False, snapToGrid=False, gridSize=0.25,
             rotateStepDegrees=90, visibilityDistance=100, fieldOfView=90, agentCount=no_robot),
        raise_for_failure=True
    )

    c.step(
        action="TeleportFull",
        position={
            "x": 1,
            "y": DEFAULT_ROBOT_HEIGHT, # scene["metadata"]["agent"]["position"]["y"],
            "z": 1,
        },
        rotation=DEFAULT_ROBOT_ROT, #scene["metadata"]["agent"]["rotation"],
        standing=True,
        horizon=30,
        forceAction=True,
    )

    # add a top view camera
    event = c.step(action="GetMapViewCameraProperties")
    event = c.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

    # maybe need to do this https://github.com/allenai/Holodeck/issues/18#issuecomment-1919531859

    # get reachabel positions

    # randomize postions of the agents  now done above
    # for i in range (no_robot):
    #    init_pos = random.choice(reachable_positions_)
    #    c.step(dict(action="Teleport", position=init_pos, agentId=i))

    return c, runtime_container, no_robot, reachable_positions