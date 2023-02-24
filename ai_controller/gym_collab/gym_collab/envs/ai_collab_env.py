import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import json_numpy

from magnebot import ActionStatus, Arm
import cv2
from aiohttp import web
from av import VideoFrame
import aiohttp_cors
import socketio
import pdb
import numpy as np
import time

from enum import Enum

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

import gymnasium as gym
from gymnasium import spaces


class AICollabEnv(gym.Env):

    #### MAIN & SETUP OF HTTP SERVER ##############################################################################


    def __init__(self, use_occupancy,view_radius, client_number, host, port, address, cert_file, key_file):

        self.pcs = set()
        self.relay = MediaRelay()
        self.tracks_received = 0

        self.frame_queue = ""

        self.client_number = client_number
        self.robot_id = 0
        self.use_occupancy = use_occupancy
        self.view_radius = view_radius
        self.centered_view = 0
        self.host = host
        self.port = port
        self.setup_ready = False


        
        

        #### SOCKET IO message function definitions ########################################################

        self.sio = socketio.Client(ssl_verify=False)

        #When first connecting
        @self.sio.event
        def connect():
            print("I'm connected!")
            if not self.use_occupancy:
                self.sio.emit("watcher_ai", (self.client_number, self.use_occupancy, "https://"+self.host+":"+str(self.port)+"/offer", 0, 0))
            else:
                self.sio.emit("watcher_ai", (self.client_number, self.use_occupancy, "", self.view_radius, self.centered_view))
            #asyncio.run(main_ai(tracks_received))

        #Receiving simulator's robot id
        @self.sio.event
        def watcher_ai(robot_id_r, occupancy_map_config):

            print("Received id", robot_id_r)
            self.robot_id = robot_id_r

            if self.use_occupancy: #When using only occupancy maps, run the main processing function here
                self.map_config = occupancy_map_config
                #asyncio.run(self.main_ai())
                self.gym_setup()
                self.setup_ready = True


        #Receiving occupancy map
        self.maps = []
        self.map_ready = False
        self.map_config = {}
        @self.sio.event
        def occupancy_map(object_type_coords_map, object_attributes_id, objects_held):

            #print("occupancy_map received")
            #s_map = json_numpy.loads(static_occupancy_map)
            c_map = json_numpy.loads(object_type_coords_map)
            self.maps = (c_map, object_attributes_id)
            self.objects_held = objects_held
            self.map_ready = True
            
            #print(c_map)


        #Connection error
        @self.sio.event
        def connect_error(data):
            print("The connection failed!")


        #Disconnect
        @self.sio.event
        def disconnect():
            print("I'm disconnected!")

        #Received a target object
        @self.sio.event
        def set_goal(agent_id,obj_id):
            print("Received new goal")
            #self.target[agent_id] = obj_id

        #Update neighbor list
        self.neighbors = {}
        self.new_neighbors = False
        @self.sio.event
        def neighbors_update(neighbors_dict):

            print('neighbors update', neighbors_dict)
            self.neighbors.update(neighbors_dict)
            self.new_neighbors = True

        #Update object list
        self.objects = {}
        self.new_objects = False
        @self.sio.event
        def objects_update(object_dict):

            
            self.objects.update(object_dict)
            
            print("objects_update", object_dict)
            self.new_objects = True
            
        #Receive messages from other agents
        self.messages = []
        @self.sio.event
        def message(message, source_agent_id):

            self.messages.append((source_agent_id,message))
            print("message", message, source_agent_id)
           
        self.new_output = () 
        @self.sio.event
        def ai_output(object_type_coords_map, object_attributes_id, objects_held, sensing_results, ai_status, extra_status, strength):
            
            self.map = json_numpy.loads(object_type_coords_map)
            self.new_output = (self.map, object_attributes_id, objects_held, sensing_results, ActionStatus(ai_status), extra_status, strength)

        #Receive status updates of our agent
        self.action_status = -1
        @self.sio.event
        def ai_status(status):

            self.action_status = ActionStatus(status)
            print("status", ActionStatus(status))


        self.run(address, cert_file, key_file)
        
        while not self.setup_ready:
            time.sleep(1)

        
    def run(self, address, cert_file, key_file):
    
        if self.use_occupancy:
            self.sio.connect(address)
            #main_thread()
        else:
            if cert_file:
                #ssl_context = ssl.SSLContext()
                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_context.load_cert_chain(cert_file, key_file)
            else:
                ssl_context = None

            app = web.Application()
            
            async def on_shutdown(app):
                # close peer connections
                coros = [pc.close() for pc in self.pcs]
                await asyncio.gather(*coros)
                self.pcs.clear()
                
            app.on_shutdown.append(on_shutdown)
            #app.router.add_get("/", index)
            #app.router.add_get("/client.js", javascript)
            app.router.add_post("/offer", self.offer)

            cors = aiohttp_cors.setup(app, defaults={
              "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*"
              )
            })

            for route in list(app.router.routes()):
                cors.add(route)

            self.sio.connect(address)
            web.run_app(
                app, access_log=None, host=self.host, port=self.port, ssl_context=ssl_context
            )



    #### GYM SETUP ######################################################################################
    
    def gym_setup(self):
    
        map_size = self.map_config['num_cells'][0]
        self.action_space = spaces.Dict(
            {
                "action" : spaces.Discrete(len(self.Action)),
                "item" : spaces.Discrete(self.map_config['num_objects']),
                "robot" : spaces.Discrete(len(self.map_config['all_robots']))
            }
        )
        
        #self.observation_space = spaces.Box(0, map_size - 1, shape=(2,), dtype=int)
        
        self.observation_space = spaces.Dict(
            {
                "frame" : spaces.Box(low=0, high=5, shape=(map_size, map_size), dtype=int),
                "objects_held" : spaces.Discrete(2),
                "action_status" : spaces.Discrete(8),
                "item_output" : spaces.Dict(
                    {
                        "item_weight" : spaces.Discrete(10),
                        "item_danger_level" : spaces.Discrete(3),
                        "item_location" : spaces.MultiDiscrete([map_size, map_size])
                    }
                ),
                "num_items" : spaces.Discrete(self.map_config['num_objects']),
                "neighbors_output" : spaces.Dict(
                    {
                        "neighbor_type" : spaces.Discrete(2),
                        "neighbor_location" : spaces.MultiDiscrete([map_size, map_size])
                    }
                
                ),
                "strength" : spaces.Discrete(len(self.map_config['all_robots'])),
                "message" : spaces.Text(min_length=0,max_length=100),
                "num_messages" : spaces.Discrete(100)
                
                #"objects_danger_level" : spaces.Box(low=1,high=2,shape=(self.map_config['num_objects'],), dtype=int)
            }
        )
        

        self.goal_count = 0
        
    
    def step(self, action):
        
        #previous_objects_held = self.objects_held
        
        world_state, sensing_output, action_terminated, action_truncated = self.take_action(action)
        #observed_state = {"frame": world_state, "message": self.messages}
        observation = {"frame": sensing_output["occupancy_map"], "objects_held": sensing_output["objects_held"], "action_status": int(action_terminated[0]) + int(action_truncated[0])*2 + int(action_terminated[1])*3 + int(action_truncated[1])*4, "num_items": len(self.object_info), "item_output": sensing_output["item_output"], "neighbors_output": sensing_output["neighbors_output"], "message": sensing_output["message"], "num_messages": len(self.messages), "strength": sensing_output["strength"] } #Occupancy map
        
        info = {}
        info['map_metadata'] = world_state[1]

        
        reward = 0
        
        #Rewards
        
        '''
        if previous_objects_held[0] and not self.objects_held[0]: #Reward given when object is left in the middle of the room
        
            goal_radius = 5
            max_x = np.round(world_state[0].shape[0]/2) + goal_radius
            min_x = np.round(world_state[0].shape[0]/2) - goal_radius
            max_y = np.round(world_state[0].shape[1]/2) + goal_radius
            min_y = np.round(world_state[0].shape[1]/2) - goal_radius
            ego_location = np.where(world_state[0] == 5)
            
            #max_x = min(ego_location[0][0] + self.view_radius, world_state[0].shape[0])
            #max_y = min(ego_location[1][0] + self.view_radius, world_state[0].shape[1])
            #min_x = max(ego_location[0][0] - self.view_radius, 0)
            #min_y = max(ego_location[1][0] - self.view_radius, 0)
            
            w_idxs = np.where(world_state[0][min_x:max_x+1,min_y:max_y+1] > 1)
            object_ids = {}
            for w_ix in range(len(w_idxs)):
                new_idx = (w_idxs[0][w_ix],w_idxs[1][w_ix])
                if previous_objects_held[0] in world_state[1][str(new_idx[0]+min_x) + str(new_idx[1]+min_y)]:
                    reward = 1
                    self.goal_count += 1
                    
        if self.objects_held[0] and not previous_objects_held[0]: #Reward given when grabbing objects
            reward = 0.5
                
        #if action_truncated: #Penalty given when not being able to grab an object or drop an object
        #    reward = -0.5
        '''
        
        
        if self.goal_count == 4: #When four objects are put in the middle the episode should terminate
            terminated = True
        else:
            terminated = False
        


        return observation, reward, terminated, False, info
        
    def reset(self, seed=None, options=None):
    
        super().reset(seed=seed)
        map_size = self.map_config['num_cells'][0]

        observation = {"frame": np.zeros((map_size,map_size),dtype=np.int64), "objects_held": 0, "action_status": 0, "num_messages": 0, "message": "", "strength": 1, "num_items": 0, "item_output": {"item_weight": 0, "item_danger_level": 0, "item_location": np.zeros((map_size,map_size), dtype=np.int64)}, "neighbors_output": {"neighbor_type": 0, "neighbor_location": np.zeros((map_size,map_size), dtype=np.int64)}}
        
        info = {}
        
        
        self.internal_state = [self.State.take_action, self.State.take_sensing_action]
        self.internal_data = {}
        self.object_info = []
        self.neighbors_info = [[um[0], 0 if um[1] == 'human' else 1,0,0] for um in self.map_config['all_robots']]
        
        self.sio.emit("reset")
        
        self.map = np.array([])
        time.sleep(2) #Sleep for reset
        
        print("Waiting for location")
        while self.map.size == 0:
            continue

        self.old_output = self.new_output
        print("Got location")
        
        observation["frame"] = self.map
        
        self.messages = []
        self.goal_count = 0
        
        return observation, info
        
    #### ROBOT API ######################################################################################

    #Forwarded magnebot API from https://github.com/alters-mit/magnebot/blob/main/doc/manual/magnebot/actions.md\

    def turn_by(self, angle, aligned_at=1):
        return ["turn_by", str(angle), "aligned_at=" + str(aligned_at)]
    def turn_to(self, target, aligned_at=1):
        return ["turn_to", str(target), "aligned_at=" + str(aligned_at)]
    def move_by(self, distance, arrived_at=0.1):
        return ["move_by", str(distance), "arrived_at=" + str(arrived_at)]
    def move_to(self, target, arrived_at=0.1, aligned_at=1, arrived_offset=0):
        return ["move_to", str(target), "arrived_at=" + str(arrived_at), "aligned_at=" + str(aligned_at), "arrived_offset="+ str(arrived_offset)]
    def reach_for(self, target, arm):
        return ["reach_for", str(target), str(arm)]
    def grasp(self, target, arm):
        return ["grasp", str(target), str(arm)]
    def drop(self, target, arm):
        return ["drop", str(target), str(arm)]
    def reset_arm(self, arm):
        return ["reset_arm", str(arm)]
    def reset_position(self):
        return ["reset_position"]
    def rotate_camera(self, roll, pitch, yaw):
        return ["rotate_camera", str(roll), str(pitch), str(yaw)]
    def look_at(self, target):
        return ["look_at", str(target)]
    def move_camera(self, position):
        return ["move_camera", str(position)]
    def reset_camera(self):
        return ["reset_camera"]
    def slide_torso(self, height):
        return ["slide_torso", str(height)]
    def danger_sensor_reading(self):
        return ["send_danger_sensor_reading"]
    def get_occupancy_map(self):
        return ["send_occupancy_map"]
    def get_objects_held_status(self):
        return ["send_objects_held_status"]


    




    #### CONTROLLER DEFINITION #####################################################################

    #Function that retrieves the newest occupancy map and makes some pre-processing if needed
    async def get_map(self, frame_queue): 


        while True:

            if self.map_ready: #Maps has all the occupancy maps and metadata
                self.map_ready = False
                await self.frame_queue.put(self.maps)
            else:
                await asyncio.sleep(0.01)
            
                
    #Function that retrieves the newest video frame and makes some pre-processing if needed
    async def get_frame(self, track,frame_queue):

        while True:
            frame = await track.recv()
            print("Processing frame")
            #frame.to_image() (av.VideoFrame)
            await self.frame_queue.put(frame)

    #Controller states
    class State(Enum):
        take_action = 1
        waiting_ongoing = 2
        grasping_object = 3
        reseting_arm = 4
        reverse_after_dropping = 5
        take_sensing_action = 6
        wait_sensing = 7
        action_end = 8
        
        
    class Action(Enum):
        move_up = 0
        move_down = 1
        move_left = 2
        move_right = 3
        move_up_right = 4
        move_up_left = 5
        move_down_right = 6
        move_down_left = 7
        grab_up = 8
        grab_right = 9
        grab_down = 10
        grab_left = 11
        grab_up_right = 12
        grab_up_left = 13
        grab_down_right = 14
        grab_down_left = 15
        drop_object = 16
        danger_sensing = 17
        get_occupancy_map = 18
        get_objects_held = 19
        check_item = 20
        check_robot = 21
        get_message = 22
        message_help_accept = 23
        message_help_request_sensing = 24
        message_help_request_lifting = 25
        message_reject_request = 26
        message_cancel_request = 27
        #check item slot 1 observation = [(item_id1, danger_level1,weight1),(item_id2, danger_level2,weight2),(item_id3, danger_level2,weight2)], number of items so far
        #get message observation text + number of messages
        #send message




    def take_action(self, action):
    
        
        terminated = False
        truncated = False
        objects_obs = []
        neighbors_obs = []
        
        
        #print(action)
            
        


        
        action_message,self.internal_state,self.internal_data,sensing_output,terminated,truncated = self.controller(action, self.old_output, self.internal_state, self.internal_data)

        if action_message: #Action message is an action to take by the robot that will be communicated to the simulator
            print("action", action_message)
            self.sio.emit("ai_action", (action_message))
                
                
        while not self.new_output: #Sync with simulator
            pass

        if action_message and any(self.new_output[5]):
            print(self.new_output[5])

        if self.new_output:
            self.old_output = self.new_output
            self.new_output = ()
        
            
        return self.old_output, sensing_output, terminated, truncated
    
    
    


    #Only works for occupancy maps not centered in magnebot
    def controller(self, complete_action, observations, internal_state, data):

        
        action_message = []
        movement_commands = 8
        grab_commands = 16
        
        occupancy_map = observations[0]
        objects_metadata = observations[1]
        objects_held = observations[2]
        danger_sensing_data = observations[3]
        action_status = observations[4]
        extra_status = observations[5]
        strength = observations[6]
        terminated = [False, False]
        truncated = [False, False]
        state = internal_state[0]
        sensing_state = internal_state[1]
        action = self.Action(complete_action["action"])
        
        sensing_output = {"occupancy_map": occupancy_map, "item_output":{"item_weight": 0, "item_danger_level": 0, "item_location": [0,0]}, "message": "", "neighbors_output":{"neighbor_type": 0, "neighbor_location": [0,0]}, "objects_held" : 0, "strength": strength}

        print(state, sensing_state)

        if state == self.State.take_action:
            if action_status != ActionStatus.ongoing:
                print("Original ", action)
                #print(occupancy_map)
            
                #self.action_status = -1
                ego_location = np.where(occupancy_map == 5)
                ego_location = np.array([ego_location[0][0],ego_location[1][0]])


                if action.value < movement_commands:
                
                    action_index = [self.Action.move_up,self.Action.move_right,self.Action.move_down,self.Action.move_left,self.Action.move_up_right,self.Action.move_up_left,self.Action.move_down_right,self.Action.move_down_left].index(action)
                    
                    original_location = np.copy(ego_location)
                    
                    ego_location = self.check_bounds(action_index, ego_location, occupancy_map)


                    if not np.array_equal(ego_location,original_location):
                        target_coordinates = np.array(self.map_config['edge_coordinate']) + ego_location*self.map_config['cell_size']
                        target = {"x": target_coordinates[0],"y": 0, "z": target_coordinates[1]}
                        state = self.State.waiting_ongoing
                        data["next_state"] = self.State.action_end
                        action_message.append(self.move_to(target=target))
                    else:
                        print("Movement not possible")
                        truncated[0] = True
                    
                elif action.value < grab_commands:    
                
                    object_location = np.copy(ego_location)
                    
                    action_index = [self.Action.grab_up,self.Action.grab_right,self.Action.grab_down,self.Action.grab_left,self.Action.grab_up_right,self.Action.grab_up_left,self.Action.grab_down_right,self.Action.grab_down_left].index(action)
                    
                    object_location = self.check_bounds(action_index, object_location, occupancy_map)
                    
                    
                    if (not np.array_equal(object_location,ego_location)) and occupancy_map[object_location[0],object_location[1]] == 2:
                        #object_location = np.where(occupancy_map == 2)
                        #key = str(object_location[0][0]) + str(object_location[1][0])
                        key = str(object_location[0]) + '_' + str(object_location[1])
                        action_message.append(self.turn_to(objects_metadata[key][0]))
                       
                        state = self.State.waiting_ongoing
                        data["next_state"] = self.State.grasping_object
                        data["object"] = objects_metadata[key][0]
                    else:
                        print("No object to grab")
                        truncated[0] = True
                    
                elif action == self.Action.drop_object:

                    if objects_held[0]:
                        action_message.append(self.drop(objects_held[0], Arm.left))
                       
                        state = self.State.waiting_ongoing
                        data["next_state"] = self.State.reverse_after_dropping

                    else:
                        print("No object to drop")
                        truncated[0] = True
                    
                
                    
                else:
                    #print("Not implemented", action)
                    pass

                
                    
                    
                
        elif state == self.State.waiting_ongoing:

            if action_status == ActionStatus.ongoing or action_status == ActionStatus.success:
                print("waiting", action_status)
                state = data["next_state"]
                    
        elif state == self.State.grasping_object:
             if action_status != ActionStatus.ongoing:
                state = self.State.waiting_ongoing
                print("waited to grasp objective")
                action_message.append(self.grasp(data["object"], Arm.left))
                del data["object"]
                data["next_state"] = self.State.reseting_arm
            
        elif state == self.State.reseting_arm:

            if action_status != ActionStatus.ongoing:
                print("waited to reset arm")
                action_message.append(self.reset_arm(Arm.left))
                state = self.State.waiting_ongoing
                data["next_state"] = self.State.action_end
                
        elif state == self.State.reverse_after_dropping:
            if action_status != ActionStatus.ongoing:
                print("waited to reverse after dropping")
                action_message.append(self.move_by(-0.5))
                state = self.State.waiting_ongoing
                data["next_state"] = self.State.action_end
                
        
                
        elif state == self.State.action_end:
            if action_status != ActionStatus.ongoing:  
                print("action end", action_status)
                terminated[0] = True
            
            
        if terminated[0] or truncated[0]:
            state = self.State.take_action
            
            
        if sensing_state == self.State.take_sensing_action:
            if action == self.Action.danger_sensing:
                action_message.append(self.danger_sensor_reading())
                sensing_state = self.State.wait_sensing
                
            elif action == self.Action.get_occupancy_map:
                action_message.append(self.get_occupancy_map())
                sensing_state = self.State.wait_sensing
            
            elif action == self.Action.get_objects_held:
                action_message.append(self.get_objects_held_status())
                sensing_state = self.State.wait_sensing
                
            elif action == self.Action.check_item:
                if complete_action["item"] >= len(self.object_info):
                    truncated[1] = True
                else:
                    sensing_output["item_output"]["item_weight"] = self.object_info[complete_action["item"]][1]
                    sensing_output["item_output"]["item_danger_level"] = self.object_info[complete_action["item"]][2]
                    sensing_output["item_output"]["item_location"] = self.object_info[complete_action["item"]][3:5]
                    terminated[1] = True
                    
            elif action == self.Action.check_robot:
            
                if complete_action["robot"] > 0:
                    robot_idx = complete_action["robot"] - 1
                    sensing_output["neighbors_output"]["neighbor_type"] =  self.neighbors_info[robot_idx][1]
                    sensing_output["neighbors_output"]["neighbor_location"] = self.neighbors_info[robot_idx][2:4]
                    terminated[1] = True
                else:
                    truncated[1] = True
                
            elif action == self.Action.get_message:
                if self.messages:
                    sensing_output["message"] = self.messages.pop(0)
                    terminated[1] = True
                else:
                    truncated[1] = True
                    
            elif action.value >= self.Action.message_help_accept.value and action.value <= self.Action.message_cancel_request.value:
            
            
                if complete_action["robot"] > 0:
            
                    robot_data = self.neighbors_info[complete_action["robot"]-1]
                    neighbors_dict = {robot_data[0]: "human" if not robot_data[1] else "ai"}
                else:
                    neighbors_dict = {robot_data[0]: "human" if not robot_data[1] else "ai" for robot_data in self.neighbors_info}
            
                if action == self.Action.message_help_accept:
                    message = "I will help "
                elif action == self.Action.message_help_request_sensing:
                    if complete_action["item"] < len(self.object_info):
                        message = "I need help with sensing " + str(self.object_info[complete_action["item"]][0])
                    else:
                        truncated[1] = True
                elif action == self.Action.message_help_request_lifting:
                    if complete_action["item"] < len(self.object_info):
                        message = "I need help with lifting " + str(self.object_info[complete_action["item"]][0])
                    else:
                        truncated[1] = True
                elif action == self.Action.message_reject_request:
                    if complete_action["robot"] > 0:
                        message = "I cannot help you right now " + str(robot_data[0])
                    else:
                        truncated[1] = True
                elif action == self.Action.message_cancel_request:
                    message = "No more need for help"
                
                
                if not truncated[1]:
                    self.sio.emit("message", (message,neighbors_dict))
                    terminated[1] = True
        
            else:
                #print("Not implemented sensing action", action)
                pass

                
        elif sensing_state == self.State.wait_sensing:
            
            if any(extra_status):
                terminated[1] = True

                if extra_status[0]: #Occupancy map received
                
                    #Update objects locations
                    object_locations = np.where(occupancy_map == 2)
                    #object_locations = np.array([object_locations[0][:],object_locations[1][:]])
                    for ol_idx in range(len(object_locations[0])):
                        key = str(object_locations[0][ol_idx]) + '_' + str(object_locations[1][ol_idx])
                        known_object = False
                        for ob_idx,ob in enumerate(self.object_info):
                            if ob[0] == objects_metadata[key][0][0]:
                                self.object_info[ob_idx][3] = object_locations[0][ol_idx]
                                self.object_info[ob_idx][4] = object_locations[1][ol_idx]
                                known_object = True
                                break
                        if not known_object:
                            self.object_info.append([objects_metadata[key][0][0],objects_metadata[key][0][1],0,object_locations[0][ol_idx],object_locations[1][ol_idx]])
                    
                    #Update robots locations
                    robots_locations = np.where(occupancy_map == 3)
                    for ol_idx in range(len(robots_locations[0])):
                        key = str(robots_locations[0][ol_idx]) + '_' + str(robots_locations[1][ol_idx])
                        for ob_idx,ob in enumerate(self.neighbors_info):
                            
                            if ob[0] == str(objects_metadata[key][0]):
                                self.neighbors_info[ob_idx][2] = robots_locations[0][ol_idx]
                                self.neighbors_info[ob_idx][3] = robots_locations[1][ol_idx]
                                break
                        
                        
                if extra_status[1]: #Danger estimate received
                    for object_key in danger_sensing_data.keys():
                        known_object = False
                        min_pos = self.map_config['edge_coordinate']
                        multiple = self.map_config['cell_size']
                        pos_new = [round((danger_sensing_data[object_key]['location'][0]+abs(min_pos))/multiple), round((danger_sensing_data[object_key]['location'][2]+abs(min_pos))/multiple)]
                        for ob_idx, ob in enumerate(self.object_info):
                            if ob[0] == object_key:
                                self.object_info[ob_idx][2] = self.combine_danger_info(danger_sensing_data[object_key]['sensor'])
                                self.object_info[ob_idx][3] = pos_new[0]
                                self.object_info[ob_idx][4] = pos_new[1]
                                known_object = True
                                break
                        if not known_object:
                            self.object_info.append([object_key,danger_sensing_data[object_key]['weight'],self.combine_danger_info(danger_sensing_data[object_key]['sensor']),pos_new[0],pos_new[1]])
               
                if extra_status[2]: #Objects held
                    sensing_output["objects_held"] = int(any(oh != 0 for oh in objects_held))
                   
                    
            
        if terminated[1] or truncated[1]:
            sensing_state = self.State.take_sensing_action    
            
            
            
        return action_message, [state,sensing_state], data, sensing_output, terminated, truncated


    def combine_danger_info(self, estimates):
        key = list(estimates.keys())[0]
        return estimates[key]['value']
        
    def check_bounds(self, action_index, location, occupancy_map):
    
        if action_index == 0: #Up
            if location[0] < occupancy_map.shape[0]-1:
                location[0] += 1
        elif action_index == 1: #Right
            if location[1] > 0:
                location[1] -= 1
        elif action_index == 2: #Down
            if location[0] > 0:
                location[0] -= 1
        elif action_index == 3: #Left
            if location[1] < occupancy_map.shape[1]-1:
                location[1] += 1
        elif action_index == 4: #Up Right
            if location[0] < occupancy_map.shape[0]-1 and location[1] > 0:
                location += [1,-1]
        elif action_index == 5: #Up Left
            if location[0] < occupancy_map.shape[0]-1 and location[1] < occupancy_map.shape[1]-1:
                location += [1,1]
        elif action_index == 6: #Down Right
            if location[0] > 0 and location[1] > 0:
                location += [-1,-1]
        elif action_index == 7: #Down Left
            if location[0] > 0 and location[1] < occupancy_map.shape[1]-1:
                location += [-1,1]
                
        return location


    #These two next functions are used to initiate the control for the robot when using only occupancy maps
    def main_thread(self):
        asyncio.run(self.main_ai())

    async def main_ai(self):


        self.gym_setup()
        #tracks_received = asyncio.Queue()
        
        self.frame_queue = asyncio.Queue()
        #print("waiting queue")
        #track = await tracks_received.get()
        print("waiting gather")
        #await asyncio.gather(get_frame(track,frame_queue),actuate(frame_queue))
        #await asyncio.gather(get_map(frame_queue),actuate(frame_queue))
        #await asyncio.gather(self.actuate(self.frame_queue))



    #### WEBRTC SETUP #####################################################################################

    #This function is used as part of the setup of WebRTC
    async def offer(self, request):


        print("offer here")
        #async def offer_async(server_id, params):
        params = await request.json()
        print(params)

        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        self.robot_id = params["id"]
        pc = RTCPeerConnection()
        pc_id = "PeerConnection(%s)" % uuid.uuid4()
        self.pcs.add(pc)

        def log_info(msg, *args):
            logger.info(pc_id + " " + msg, *args)

        log_info("Created for %s", request.remote)

        # prepare local media

        if args.record_to:
            recorder = MediaRecorder(args.record_to)
        else:
            recorder = MediaBlackhole()



        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

       

        @pc.on("track")
        async def on_track(track):

            log_info("Track %s received", track.kind)


            if track.kind == "video":


                if args.record_to:
                    print("added record")
                    recorder.addTrack(self.relay.subscribe(track))

                if not self.tracks_received:
                    #processing_thread = threading.Thread(target=main_thread, args = (track, ))
                    #processing_thread.daemon = True
                    #processing_thread.start()
                    self.frame_queue = asyncio.Queue()
                    #print("waiting queue")
                    #track = await tracks_received.get()
                    print("waiting gather")
                    self.tracks_received += 1
                    await asyncio.gather(self.get_frame(track,self.frame_queue),self.actuate(self.frame_queue))
                #tracks_received.append(relay.subscribe(track))
                
                #print(tracks_received.qsize())
            
                
                

            @track.on("ended")
            async def on_ended():
                log_info("Track %s ended", track.kind)
                await recorder.stop()

        # handle offer
        await pc.setRemoteDescription(offer)
        await recorder.start()

        # send answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        print("offer",json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}))
        
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )








if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--use-occupancy", action='store_true', help="Use occupancy maps instead of images")
    parser.add_argument("--address", default='https://172.17.15.69:4000', help="Address where our simulation is running")
    parser.add_argument("--robot-number", default=1, help="Robot number to control")
    parser.add_argument("--view-radius", default=0, help="When using occupancy maps, the view radius")

    args = parser.parse_args()
    
    logger = logging.getLogger("pc")

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    aicollab = AICollabEnv(args.use_occupancy,args.view_radius, int(args.robot_number), args.host, args.port, args.address,args.cert_file, args.key_file)
    #aicollab.run(args.address,args.cert_file, args.key_file)
    #print("Finished here")
    #while not aicollab.setup_ready:
    #    time.sleep(1)
    aicollab.step(0)