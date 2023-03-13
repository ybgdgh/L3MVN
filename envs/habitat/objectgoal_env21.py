import json
import bz2
import gzip
import _pickle as cPickle
import gym
import numpy as np
import quaternion
import skimage.morphology
import habitat

from envs.utils.fmm_planner import FMMPlanner
from constants import category_to_id, mp3d_category_id
import envs.utils.pose as pu

coco_categories = [0, 3, 2, 4, 5, 1]

class ObjectGoal_Env21(habitat.RLEnv):
    """The Object Goal Navigation environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, args, rank, config_env, dataset):
        self.args = args
        self.rank = rank

        super().__init__(config_env, dataset)

        # Initializations
        self.episode_no = 0

        # Scene info
        self.last_scene_path = None
        self.scene_path = None
        self.scene_name = None

        # Episode Dataset info
        self.eps_data = None
        self.eps_data_idx = None
        self.gt_planner = None
        self.object_boundary = None
        self.goal_idx = None
        self.goal_name = None
        self.map_obj_origin = None
        self.starting_loc = None
        self.starting_distance = None

        # Episode tracking info
        self.curr_distance = None
        self.prev_distance = None
        self.timestep = None
        self.stopped = None
        self.path_length = None
        self.last_sim_location = None
        self.trajectory_states = []
        self.info = {}
        self.info['distance_to_goal'] = None
        self.info['spl'] = None
        self.info['success'] = None

        # self.scene = self._env.sim.semantic_annotations()

        fileName = 'data/matterport_category_mappings.tsv'

        text = ''
        lines = []
        items = []
        self.hm3d_semantic_mapping={}

        with open(fileName, 'r') as f:
            text = f.read()
        lines = text.split('\n')

        for l in lines:
            items.append(l.split('    '))

        for i in items:
            if len(i) > 3:
                self.hm3d_semantic_mapping[i[2]] = i[-1]

        # print()

        # for obj in self.scene.objects:
        #     if obj is not None:
        #         print(
        #             f"Object id:{obj.id}, category:{obj.category.name()}, index:{obj.category.index()}"
        #             f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
        #         )

    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        args = self.args
        # new_scene = self.episode_no % args.num_train_episodes == 0


        self.episode_no += 1

        # Initializations
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.trajectory_states = []

        # if new_scene:
        obs = super().reset()
        start_height = 0
        self.scene = self._env.sim.semantic_annotations()
        # start_height = self._env.current_episode.start_position[1]
        # goal_height = self.scene.objects[self._env.current_episode.info['closest_goal_object_id']].aabb.center[1]

        # floor_height = []
        # floor_size = []
        # for obj in self.scene.objects:
        #     if obj.category.name() in self.hm3d_semantic_mapping and \
        #         self.hm3d_semantic_mapping[obj.category.name()] == 'floor':
        #         floor_height.append(abs(obj.aabb.center[1] - start_height))
        #         floor_size.append(obj.aabb.sizes[0]*obj.aabb.sizes[2])

        
        # if not args.eval:
        #     while all(h > 0.1 or s < 12 for (h,s) in zip(floor_height, floor_size)) or abs(start_height-goal_height) > 1.2:
        #         obs = super().reset()

        #         self.scene = self._env.sim.semantic_annotations()
        #         start_height = self._env.current_episode.start_position[1]
        #         goal_height = self.scene.objects[self._env.current_episode.info['closest_goal_object_id']].aabb.center[1]

        #         floor_height = []
        #         floor_size = []
        #         for obj in self.scene.objects:
        #             if obj.category.name() in self.hm3d_semantic_mapping and \
        #                 self.hm3d_semantic_mapping[obj.category.name()] == 'floor':
        #                 floor_height.append(abs(obj.aabb.center[1] - start_height))
        #                 floor_size.append(obj.aabb.sizes[0]*obj.aabb.sizes[2])

        self.prev_distance = self._env.get_metrics()["distance_to_goal"]
        self.starting_distance = self._env.get_metrics()["distance_to_goal"]


        # print("obs: ,", obs)
        # print("obs shape: ,", obs.shape)
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        semantic = self._preprocess_semantic(obs["semantic"])
        # print("rgb shape: ,", rgb.shape)
        # print("depth shape: ,", depth.shape)
        # print("semantic shape: ,", semantic.shape)

        state = np.concatenate((rgb, depth, semantic), axis=2).transpose(2, 0, 1)
        self.last_sim_location = self.get_sim_location()

        # Set info
        self.info['time'] = self.timestep
        self.info['sensor_pose'] = [0., 0., 0.]
        self.info['goal_cat_id'] = coco_categories[obs['objectgoal'][0]]
        self.info['goal_name'] = category_to_id[obs['objectgoal'][0]]

        self.goal_name = category_to_id[obs['objectgoal'][0]]

        return state, self.info

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        action = action["action"]
        if action == 0:
            self.stopped = True
            # Not sending stop to simulator, resetting manually
            action = 3

        obs, rew, done, _ = super().step(action)

        # Get pose change
        dx, dy, do = self.get_pose_change()
        self.info['sensor_pose'] = [dx, dy, do]
        self.path_length += pu.get_l2_distance(0, dx, 0, dy)

        spl, success, dist = 0., 0., 0.
        if done:
            spl, success, dist = self.get_metrics()
            self.info['distance_to_goal'] = dist
            self.info['spl'] = spl
            self.info['success'] = success

        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        semantic = self._preprocess_semantic(obs["semantic"])
        state = np.concatenate((rgb, depth, semantic), axis=2).transpose(2, 0, 1)

        self.timestep += 1
        self.info['time'] = self.timestep

        return state, rew, done, self.info

    def _preprocess_semantic(self, semantic):
        # print("*********semantic type: ", type(semantic))
        se = list(set(semantic.ravel()))
        # print(se) # []
        for i in range(len(se)):
            if self.scene.objects[se[i]].category.name() in self.hm3d_semantic_mapping:
                hm3d_category_name = self.hm3d_semantic_mapping[self.scene.objects[se[i]].category.name()]
            else:
                hm3d_category_name = self.scene.objects[se[i]].category.name()

            if hm3d_category_name in mp3d_category_id:
                # print("sum: ", np.sum(sem_output[sem_output==se[i]])/se[i])
                semantic[semantic==se[i]] = mp3d_category_id[hm3d_category_name]-1
            else :
                semantic[
                    semantic==se[i]
                    ] = 0
    
        # se = list(set(semantic.ravel()))
        # print("semantic: ", se) # []
        # semantic = np.expand_dims(semantic.astype(np.uint8), 2)
        return semantic.astype(np.uint8)

    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0., 1.0)

    def get_reward(self, observations):
        self.curr_distance = self._env.get_metrics()['distance_to_goal']

        reward = (self.prev_distance - self.curr_distance) * \
            self.args.reward_coeff

        self.prev_distance = self.curr_distance
        return reward

    def get_llm_distance(self, target_point_map, frontier_loc_g):

        frontier_dis_g = self.gt_planner.fmm_dist[frontier_loc_g[0],
                                                frontier_loc_g[1]] / 20.0
        tpm = len(list(set(target_point_map.ravel()))) -1
        for lay in range(tpm):
            frontier_loc = np.argwhere(target_point_map == lay+1)
            frontier_distance = self.gt_planner.fmm_dist[frontier_loc[0],
                                                      frontier_loc[1]] / 20.0
            if frontier_distance > frontier_dis_g:
                return 0
        return 1
        
    def get_metrics(self):
        """This function computes evaluation metrics for the Object Goal task

        Returns:
            spl (float): Success weighted by Path Length
                        (See https://arxiv.org/pdf/1807.06757.pdf)
            success (int): 0: Failure, 1: Successful
            dist (float): Distance to Success (DTS),  distance of the agent
                        from the success threshold boundary in meters.
                        (See https://arxiv.org/pdf/2007.00643.pdf)
        """
        dist = self._env.get_metrics()['distance_to_goal']
        if dist < 0.1:
            success = 1
        else:
            success = 0
        spl = min(success * self.starting_distance / self.path_length, 1)
        return spl, success, dist

    def get_done(self, observations):
        if self.info['time'] >= self.args.max_episode_length - 1:
            done = True
        elif self.stopped:
            done = True
            # print(self._env.get_metrics())
        else:
            done = False
        return done

    def _episode_success(self):
        return self._env.get_metrics()['success']

    def get_info(self, observations):
        """This function is not used, Habitat-RLEnv requires this function"""
        info = {}
        return info

    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do
