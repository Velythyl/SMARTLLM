import ai2thor.controller

from ai2holodeck.constants import THOR_COMMIT_ID
from hippo.ai2thor_hippo_controller import get_hippo_controller
from hippo.utils.file_utils import get_tmp_folder

TARGET_TMP_DIR = get_tmp_folder()

def get_controller(scene, **kwargs):

    if "target_dir" in kwargs:
        target_dir = kwargs.pop("target_dir")
    else:
        target_dir = TARGET_TMP_DIR
    controller = get_hippo_controller(scene, target_dir=target_dir, **kwargs)
    return controller

    if isinstance(scene, str):
        controller = ai2thor.controller.Controller(commit_id=THOR_COMMIT_ID, scene=scene, **kwargs)
    else:
        assert isinstance(scene, dict)
        if "target_dir" in kwargs:
            target_dir = kwargs.pop("target_dir")
        else:
            target_dir = TARGET_TMP_DIR
        controller = get_hippo_controller(scene, target_dir=target_dir, **kwargs)
    return controller
