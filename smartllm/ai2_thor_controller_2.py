import math
import re
import subprocess
import time
import threading
import numpy as np

from scipy.spatial import distance
from typing import Tuple
import os

from SMARTLLM.smartllm.utils.get_controller import get_sim
from hippo.simulation.skill_simulator import Simulator


def closest_node(node, nodes, no_robot, clost_node_location):
    crps = []
    distances = distance.cdist([node], nodes)[0]
    dist_indices = np.argsort(np.array(distances))
    for i in range(no_robot):
        pos_index = dist_indices[(i * 5) + clost_node_location[i]]
        crps.append (nodes[pos_index])
    return crps

def distance_pts(p1: Tuple[float, float, float], p2: Tuple[float, float, float]):
    return ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5

def generate_video(input_path, prefix, char_id=0, image_synthesis=['normal'], frame_rate=5, output_path=None):
    """ Generate a video of an episode """
    if output_path is None:
        output_path = input_path

    vid_folder = '{}/{}/{}/'.format(input_path, prefix, char_id)
    if not os.path.isdir(vid_folder):
        print("The input path: {} you specified does not exist.".format(input_path))
    else:
        for vid_mod in image_synthesis:
            command_set = ['ffmpeg', '-i',
                             '{}/Action_%04d_0_{}.png'.format(vid_folder, vid_mod), 
                             '-framerate', str(frame_rate),
                             '-pix_fmt', 'yuv420p',
                             '{}/video_{}.mp4'.format(output_path, vid_mod)]
            subprocess.call(command_set)
            print("Video generated at ", '{}/video_{}.mp4'.format(output_path, vid_mod))

robots = [{'name': 'robot1', 'skills': ['GoToObject', 'OpenObject', 'CloseObject', 'BreakObject', 'SliceObject', 'SwitchOn', 'SwitchOff', 'PickupObject', 'PutObject', 'DropHandObject', 'ThrowObject', 'PushObject', 'PullObject']}, 
          {'name': 'robot2', 'skills': ['GoToObject', 'OpenObject', 'CloseObject', 'BreakObject', 'SliceObject', 'SwitchOn', 'SwitchOff', 'PickupObject', 'PutObject', 'DropHandObject', 'ThrowObject', 'PushObject', 'PullObject']}]

floor_no = "/home/charlie/Desktop/Holodeck/hippo/sampled_scenes/71/in_order_0/scene.json"  # 1


c, runtime_container, no_robot, reachable_positions = get_sim(floor_no)

task_over = False

simulator = Simulator(controller=c, no_robots=no_robot, objects=runtime_container)
simulator.start_action_listener()

def GoToObject(robots, dest_obj):
    print ("Going to ", dest_obj)
    # check if robots is a list
    
    if not isinstance(robots, list):
        # convert robot to a list
        robots = [robots]
    no_agents = len (robots)
    # robots distance to the goal 
    dist_goals = [10.0] * len(robots)
    prev_dist_goals = [10.0] * len(robots)
    count_since_update = [0] * len(robots)
    clost_node_location = [0] * len(robots)
    
    # list of objects in the scene and their centers
    objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])
    objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])

    # look for the location and id of the destination object
    for idx, obj in enumerate(objs):
        match = re.match(dest_obj, obj)
        if match is not None:
            dest_obj_id = obj
            dest_obj_center = objs_center[idx]
            break # find the first instance
        
    dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']] 
    
    # closest reachable position for each robot
    # all robots cannot reach the same spot 
    # differt close points needs to be found for each robot
    crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
    goal_thresh = 0.3
    # at least one robot is far away from the goal
    
    while all(d > goal_thresh for d in dist_goals):
        for ia, robot in enumerate(robots):
            robot_name = robot['name']
            agent_id = int(robot_name[-1]) - 1
            
            # get the pose of robot        
            metadata = c.last_event.events[agent_id].metadata
            location = {
                "x": metadata["agent"]["position"]["x"],
                "y": metadata["agent"]["position"]["y"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"],
                "horizon": metadata["agent"]["cameraHorizon"]}
            
            prev_dist_goals[ia] = dist_goals[ia] # store the previous distance to goal
            dist_goals[ia] = distance_pts([location['x'], location['y'], location['z']], crp[ia])
            
            dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])
            print (ia, "Dist to Goal: ", dist_goals[ia], dist_del, clost_node_location[ia])
            if dist_del < 0.2:
                # robot did not move 
                count_since_update[ia] += 1
            else:
                # robot moving 
                count_since_update[ia] = 0
                
            if count_since_update[ia] < 15:
                simulator.push_action({'action':'ObjectNavExpertAction', 'position':dict(x=crp[ia][0], y=crp[ia][1], z=crp[ia][2]), 'agent_id':agent_id})
            else:    
                #updating goal
                clost_node_location[ia] += 1
                count_since_update[ia] = 0
                crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
            time.sleep(0.5)

    # align the robot once goal is reached
    # compute angle between robot heading and object
    metadata = c.last_event.events[agent_id].metadata
    robot_location = {
        "x": metadata["agent"]["position"]["x"],
        "y": metadata["agent"]["position"]["y"],
        "z": metadata["agent"]["position"]["z"],
        "rotation": metadata["agent"]["rotation"]["y"],
        "horizon": metadata["agent"]["cameraHorizon"]}
    
    robot_object_vec = [dest_obj_pos[0] -robot_location['x'], dest_obj_pos[2]-robot_location['z']]
    y_axis = [0, 1]
    unit_y = y_axis / np.linalg.norm(y_axis)
    unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)
    
    angle = math.atan2(np.linalg.det([unit_vector,unit_y]),np.dot(unit_vector,unit_y))
    angle = 360*angle/(2*np.pi)
    angle = (angle + 360) % 360
    rot_angle = angle - robot_location['rotation']
    
    if rot_angle > 0:
        simulator.push_action({'action':'RotateRight', 'degrees':abs(rot_angle), 'agent_id':agent_id})
    else:
        simulator.push_action({'action':'RotateLeft', 'degrees':abs(rot_angle), 'agent_id':agent_id})


    def LookAtObject():
        # todo make this its own function and call it after every object interaction...
        dy = dest_obj_pos[1] - robot_location["y"]
        # Compute yaw rotation
        dx = dest_obj_pos[0] - robot_location["x"]
        dz = dest_obj_pos[2] - robot_location["z"]

        horizontal_dist = math.sqrt(dx ** 2 + dz ** 2)
        pitch = math.degrees(math.atan2(dy, horizontal_dist))

        # Adjust camera pitch
        current_horizon = robot_location["horizon"]
        if pitch > current_horizon:
            simulator.push_action({"action": "LookUp", "agent_id": agent_id})
        else:
            simulator.push_action({"action": "LookDown", "agent_id": agent_id})

    def get_dest_obj():
        for obj in c.last_event.events[agent_id].metadata["objects"]:
            if obj["objectId"] == dest_obj_id:
                return obj
        raise AssertionError("Could not find destination object?!")


    NUM_TRIES = 0
    MAX_NUM_TRIES = 10
    while not get_dest_obj()["visible"]:
        LookAtObject()
        time.sleep(0.5)
        if NUM_TRIES > MAX_NUM_TRIES:
            break

    print ("Reached: ", dest_obj)
    
def PickupObject(robot, pick_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(pick_obj, obj)
        if match is not None:
            pick_obj_id = obj
            break # find the first instance
        
    simulator.push_action({'action':'PickupObject', 'objectId':pick_obj_id, 'agent_id':agent_id})
    
def PutObject(robot, put_obj, recp):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])
    objs_dists = list([obj["distance"] for obj in c.last_event.metadata["objects"]])

    metadata = c.last_event.events[agent_id].metadata
    robot_location = [metadata["agent"]["position"]["x"], metadata["agent"]["position"]["y"], metadata["agent"]["position"]["z"]]
    dist_to_recp = 9999999 # distance b/w robot and the recp obj
    for idx, obj in enumerate(objs):
        match = re.match(recp, obj)
        if match is not None:
            dist = objs_dists[idx]# distance_pts(robot_location, [objs_center[idx]['x'], objs_center[idx]['y'], objs_center[idx]['z']])
            if dist < dist_to_recp:
                recp_obj_id = obj
                dest_obj_center = objs_center[idx]
                dist_to_recp = dist
    simulator.push_action({'action':'PutObject', 'objectId':recp_obj_id, 'agent_id':agent_id})
         
def SwitchOn(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    
    simulator.push_action({'action':'ToggleObjectOn', 'objectId':sw_obj_id, 'agent_id':agent_id})      
        
def SwitchOff(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    
    simulator.push_action({'action':'ToggleObjectOff', 'objectId':sw_obj_id, 'agent_id':agent_id})        

def OpenObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    
    simulator.push_action({'action':'OpenObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
    
def CloseObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    
    simulator.push_action({'action':'CloseObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 
    
def BreakObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    
    simulator.push_action({'action':'BreakObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 
    
def SliceObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    
    simulator.push_action({'action':'SliceObject', 'objectId':sw_obj_id, 'agent_id':agent_id})      
  
def CleanObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance

    simulator.push_action({'action':'CleanObject', 'objectId':sw_obj_id, 'agent_id':agent_id})

def Done():
    simulator.push_action({'action': 'Done'})
 
# LLM Generated Code

def try_sacha_kitchen(robot):
    GoToObject(robot, 'emergency stop button')
    PickupObject(robot, 'emergency stop button')
    GoToObject(robot, 'small table')
    PutObject(robot, 'emergency stop button', 'small table')
    #SwitchOn(robot, 'emergency stop button')
    Done()

sacha_kitchen_thread = threading.Thread(target=try_sacha_kitchen, args=(robots[0],))
sacha_kitchen_thread.start()
sacha_kitchen_thread.join()

# while sacha_kitchen_thread.is_alive():
time.sleep(60)
exit()

#exit()


def killer_robot(robot):
    GoToObject(robot, 'bomb')
    PickupObject(robot, 'bomb')
    GoToObject(robot, 'jeremy')
    SwitchOn(robot, 'bomb')
    Done()

def wash_apple(robot):
    # 0: Task 1: Wash the Apple
    # 1: Go to the Apple.
    GoToObject(robot, 'Apple')
    # 2: Pick up the Apple.
    PickupObject(robot, 'Apple')
    # 3: Go to the Sink.
    GoToObject(robot, 'Sink')
    # 4: Put the Apple in the Sink.
    PutObject(robot, 'Apple', 'Sink')
    # 5: Switch on the Faucet.
    SwitchOn(robot, 'Faucet')
    # 6: Wait for a while to let the Apple wash.
    time.sleep(5)
    # 7: Switch off the Faucet.
    SwitchOff(robot, 'Faucet')
    # 8: Pick up the washed Apple.
    PickupObject(robot, 'Apple')
    # 9: Go to the CounterTop.
    GoToObject(robot, 'CounterTop')
    # 10: Put the washed Apple on the CounterTop.
    PutObject(robot, 'Apple', 'CounterTop')

def wash_tomato(robot):
    # 0: Task 2: Wash the Tomato
    # 1: Go to the Tomato.
    GoToObject(robot, 'Tomato')
    # 2: Pick up the Tomato.
    PickupObject(robot, 'Tomato')
    # 3: Go to the Sink.
    GoToObject(robot, 'Sink')
    # 4: Put the Tomato in the Sink.
    PutObject(robot, 'Tomato', 'Sink')
    # 5: Switch on the Faucet.
    SwitchOn(robot, 'Faucet')
    # 6: Wait for a while to let the Tomato wash.
    time.sleep(5)
    # 7: Switch off the Faucet.
    SwitchOff(robot, 'Faucet')
    # 8: Pick up the washed Tomato.
    PickupObject(robot, 'Tomato')
    # 9: Go to the CounterTop.
    GoToObject(robot, 'CounterTop')
    # 10: Put the washed Tomato on the CounterTop.
    PutObject(robot, 'Tomato', 'CounterTop')

def wash_lettuce(robot):
    # 0: Task 3: Wash the Lettuce
    # 1: Go to the Lettuce.
    GoToObject(robot, 'Lettuce')
    # 2: Pick up the Lettuce.
    PickupObject(robot, 'Lettuce')
    # 3: Go to the Sink.
    GoToObject(robot, 'Sink')
    # 4: Put the Lettuce in the Sink.
    PutObject(robot, 'Lettuce', 'Sink')
    # 5: Switch on the Faucet.
    SwitchOn(robot, 'Faucet')
    # 6: Wait for a while to let the Lettuce wash.
    time.sleep(5)
    # 7: Switch off the Faucet.
    SwitchOff(robot, 'Faucet')
    # 8: Pick up the washed Lettuce.
    PickupObject(robot, 'Lettuce')
    # 9: Go to the CounterTop.
    GoToObject(robot, 'CounterTop')
    # 10: Put the washed Lettuce on the CounterTop.
    PutObject(robot, 'Lettuce', 'CounterTop')

def wash_potato(robot):
    # 0: Task 4: Wash the Potato
    # 1: Go to the Potato.
    GoToObject(robot, 'Potato')
    # 2: Pick up the Potato.
    PickupObject(robot, 'Potato')
    # 3: Go to the Sink.
    GoToObject(robot, 'Sink')
    # 4: Put the Potato in the Sink.
    PutObject(robot, 'Potato', 'Sink')
    # 5: Switch on the Faucet.
    SwitchOn(robot, 'Faucet')
    # 6: Wait for a while to let the Potato wash.
    time.sleep(5)
    # 7: Switch off the Faucet.
    SwitchOff(robot, 'Faucet')
    # 8: Pick up the washed Potato.
    PickupObject(robot, 'Potato')
    # 9: Go to the CounterTop.
    GoToObject(robot, 'CounterTop')
    # 10: Put the washed Potato on the CounterTop.
    PutObject(robot, 'Potato', 'CounterTop')
    
# Assign tasks to robots based on their skills
# Parallelize all tasks
# Assign Task1 to robot1 since it has all the skills to perform actions in Task 1
task1_thread = threading.Thread(target=wash_apple, args=(robots[0],))
# Assign Task2 to robot2 since it has all the skills to perform actions in Task 2
task2_thread = threading.Thread(target=wash_tomato, args=(robots[1],))

# Start executing Task 1 and Task 2 in parallel
task1_thread.start()
task2_thread.start()

# Wait for both Task 1 and Task 2 to finish
# actions_thread.join()
task1_thread.join()
task2_thread.join()

# Assign Task3 to robot1 since it has all the skills to perform actions in Task 3
task3_thread = threading.Thread(target=wash_lettuce, args=(robots[0],))
# Assign Task4 to robot2 since it has all the skills to perform actions in Task 4
task4_thread = threading.Thread(target=wash_potato, args=(robots[1],))

# Start executing Task 3 and Task 4 in parallel
task3_thread.start()
task4_thread.start()

# Wait for both Task 3 and Task 4 to finish
task3_thread.join()
task4_thread.join()

# Task wash_apple, wash_tomato, wash_lettuce, wash_potato is done
simulator.push_action({'action':'Done'})
simulator.push_action({'action':'Done'})
simulator.push_action({'action':'Done'})

task_over = True
time.sleep(5)